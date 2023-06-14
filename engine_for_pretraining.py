import math
import sys
from typing import Iterable
import torch
import torch.nn as nn
import utils
from einops import rearrange
from transformers import BertTokenizerFast
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from utils import mask_batch_text_tokens, NormSoftmaxLoss

def train_one_epoch(model: torch.nn.Module, data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0, patch_size: int = 16, 
                    normlize_target: bool = True, log_writer=None, lr_scheduler=None, start_steps=None,
                    lr_schedule_values=None, wd_schedule_values=None, now_epoch: int = 0, _mlm_probability: int = 0.15):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')


    loss_func_v = nn.MSELoss()
    loss_func_t = nn.CrossEntropyLoss()
    loss_func_vtc = NormSoftmaxLoss()
    loss_func_vtm = nn.CrossEntropyLoss()
    for step, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # assign learning rate & weight decay for each step
        it = start_steps + step  # global training iteration
        if lr_schedule_values is not None or wd_schedule_values is not None:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        #===============================================================================================
        videos, bool_masked_pos,text_str = batch['process_data'], batch['mask'], batch['text']
        videos = videos.to(device, non_blocking=True)
        bool_masked_pos = bool_masked_pos.to(device, non_blocking=True).flatten(1).to(torch.bool)
        text_token = tokenizer(text_str, return_tensors='pt', padding=True, truncation=True, max_length=512)
        #================================== VTM ==========================================================
        # in model
        #=================================== MLM =========================================================
        mlm_flag = True
        if mlm_flag:
            input_ids_no_mlm = text_token.input_ids.clone()
            #text_token.input_ids has been changed
            text_mlm_input_ids, mlm_labels = mask_batch_text_tokens(text_token.input_ids, tokenizer, is_train=True, mlm_probability=_mlm_probability)
            text_token.update({'mlm_input_ids':text_mlm_input_ids}) 
            text_token.update({'input_ids_no_mlm':input_ids_no_mlm}) 
        text_token = {key: val.to(device) for key, val in text_token.items()}
        mlm_labels = mlm_labels.to(device)
        #===============================================================================================


        with torch.no_grad():
            # calculate the predict label
            mean = torch.as_tensor(IMAGENET_DEFAULT_MEAN).to(device)[None, :, None, None, None]
            std = torch.as_tensor(IMAGENET_DEFAULT_STD).to(device)[None, :, None, None, None]
            unnorm_videos = videos * std + mean  # in [0, 1]

            if normlize_target:
                videos_squeeze = rearrange(unnorm_videos, 'b c (t p0) (h p1) (w p2) -> b (t h w) (p0 p1 p2) c', p0=2, p1=patch_size, p2=patch_size)
                videos_norm = (videos_squeeze - videos_squeeze.mean(dim=-2, keepdim=True)
                    ) / (videos_squeeze.var(dim=-2, unbiased=True, keepdim=True).sqrt() + 1e-6)
                # we find that the mean is about 0.48 and standard deviation is about 0.08.
                videos_patch = rearrange(videos_norm, 'b n p c -> b n (p c)')
            else:
                videos_patch = rearrange(unnorm_videos, 'b c (t p0) (h p1) (w p2) -> b (t h w) (p0 p1 p2 c)', p0=2, p1=patch_size, p2=patch_size)

            B, _, C = videos_patch.shape
            labels = videos_patch[bool_masked_pos].reshape(B, -1, C)


        x_vis_f, mlm_logits ,vtm_logits, vtm_labels, t_feature, v_feature  = model(videos, bool_masked_pos, text_token)

        loss_v = loss_func_v(input=x_vis_f, target=labels)
        loss_t = loss_func_t(input=mlm_logits.view(-1, 30522), target=mlm_labels.view(-1))
        loss_vtc = loss_func_vtc(sim_matrix(t_feature, v_feature))
        loss_vtm = loss_func_vtm(input=vtm_logits, target=vtm_labels.long())
        loss = loss_v + loss_t + loss_vtc + loss_vtm
        loss_value = loss.item()
        loss_value_v = loss_v.item()
        loss_value_t = loss_t.item()
        loss_value_vtc = loss_vtc.item()
        loss_value_vtm = loss_vtm.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()
        # compute gradient and do SGD step
        loss.backward()
        optimizer.step()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        metric_logger.update(loss_v=loss_value_v)
        metric_logger.update(loss_t=loss_value_t)
        metric_logger.update(loss_vtc=loss_value_vtc)
        metric_logger.update(loss_vtm=loss_value_vtm)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)

        if log_writer is not None:           
            log_writer.update(loss=loss_value, head="loss")
            log_writer.update(loss_v=loss_value_v, head="loss")
            log_writer.update(loss_t=loss_value_t, head="loss")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            log_writer.set_step()

        if lr_scheduler is not None:
            lr_scheduler.step_update(start_steps + step)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if log_writer is not None:
        log_writer.update(epoch=now_epoch, head="loss")
        log_writer.update(avg_loss=metric_logger.meters['loss'].global_avg, head="loss")
        log_writer.set_step()
        
    torch.cuda.empty_cache()
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}



def sim_matrix(a, b, eps=1e-8):
    """
    added eps for numerical stability
    """
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt

    