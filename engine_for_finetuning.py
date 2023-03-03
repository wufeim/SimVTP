import os
import numpy as np
import math
import sys
from typing import Iterable, Optional
import torch
from timm.data import Mixup
from timm.utils import accuracy, ModelEma
import utils
from loss import NormSoftmaxLoss
import torch.distributed as dist

def sim_matrix(a, b, eps=1e-8):
    """
    added eps for numerical stability
    """
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt


class AllGather(torch.autograd.Function):
    """An autograd function that performs allgather on a tensor."""

    @staticmethod
    def forward(ctx, tensor, n_gpu):
        output = [torch.zeros_like(tensor) for _ in range(n_gpu)]
        dist.all_gather(output, tensor)
        torch.distributed.barrier()
        return torch.cat(output, 0)


def train_class_batch(model, samples, texts, criterion):
    video_embedding,text_embeddings = model(samples, texts)
    output = sim_matrix(text_embeddings, video_embedding)
    loss = criterion(output)
    return loss, output


def get_loss_scale_for_deepspeed(model):
    optimizer = model.optimizer
    return optimizer.loss_scale if hasattr(optimizer, "loss_scale") else optimizer.cur_scale


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, log_writer=None,
                    start_steps=None, lr_schedule_values=None, wd_schedule_values=None,
                    num_training_steps_per_epoch=None, update_freq=None, tokenizer=None):
    model.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
    
    if loss_scaler is None:
        model.zero_grad()
        model.micro_steps = 0
    else:
        optimizer.zero_grad()

    for data_iter_step, (samples, texts, _, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        step = data_iter_step // update_freq
        if step >= num_training_steps_per_epoch:
            continue
        it = start_steps + step  # global training iteration
        # Update LR & WD for the first acc
        if lr_schedule_values is not None or wd_schedule_values is not None and data_iter_step % update_freq == 0:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        
        samples = samples.to(device, non_blocking=True)
        texts = tokenizer(texts, return_tensors='pt', padding=True,truncation=True)
        texts = {key: val.to(device) for key, val in texts.items()}

        if loss_scaler is None:
            samples = samples.half()
            loss, output = train_class_batch(
                model, samples, texts, criterion)
        else:
            with torch.cuda.amp.autocast():
                loss, output = train_class_batch(
                    model, samples, texts, criterion)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        if loss_scaler is None:
            loss /= update_freq
            model.backward(loss)
            model.step()

            if (data_iter_step + 1) % update_freq == 0:
                # Deepspeed will call step() & model.zero_grad() automatic
                if model_ema is not None:
                    model_ema.update(model)
            grad_norm = None
            loss_scale_value = get_loss_scale_for_deepspeed(model)
        else:
            # this attribute is added by timm on one optimizer (adahessian)
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            loss /= update_freq
            grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                    parameters=model.parameters(), create_graph=is_second_order,
                                    update_grad=(data_iter_step + 1) % update_freq == 0)
            if (data_iter_step + 1) % update_freq == 0:
                optimizer.zero_grad()
                if model_ema is not None:
                    model_ema.update(model)
            loss_scale_value = loss_scaler.state_dict()["scale"]

        torch.cuda.synchronize()



        metric_logger.update(loss=loss_value)
        metric_logger.update(loss_scale=loss_scale_value)
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
        metric_logger.update(grad_norm=grad_norm)
        
        if log_writer is not None:
            log_writer.update(loss=loss_value, head="loss")
            log_writer.update(loss_scale=loss_scale_value, head="opt")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            log_writer.update(grad_norm=grad_norm, head="opt")

            log_writer.set_step()

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if log_writer is not None:
        log_writer.update(epoch=epoch, head="loss")
        log_writer.update(avg_loss=metric_logger.meters['loss'].global_avg, head="loss")
        log_writer.set_step()
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def validation_one_epoch(args, data_loader, model, device, tokenizer, evl_metrics, global_rank, epoch):
    criterion = NormSoftmaxLoss()
    allgather = AllGather.apply
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Val:'

    # switch to evaluation mode
    model.eval()

    video_embeds_list = []
    text_embeds_list = []
    name_list = []
    n_gpu = utils.get_world_size()
    print('n_gpu :', n_gpu )
    test_num_segment = args.test_num_segment
    test_num_crop = args.test_num_crop
    for batch in metric_logger.log_every(data_loader, 10, header):
        videos = batch[0]
        texts = batch[1]
        ids = batch[2]
        chunk_nb = batch[3].to(device, non_blocking=True)
        split_nb = batch[4].to(device, non_blocking=True)
        
        id_tensor = torch.tensor([int(i) for i in ids]).to(device, non_blocking=True)
        info = torch.stack([id_tensor,chunk_nb,split_nb],dim=1)

        videos = videos.to(device, non_blocking=True)
        texts = tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
        texts = {key: val.to(device, non_blocking=True) for key, val in texts.items()}
 
        # compute output
        with torch.cuda.amp.autocast():
            video_embedding,text_embeddings = model(videos, texts)

            video_embedding_with_info = torch.cat([video_embedding,info],dim=1)
            text_embeddings_with_info = torch.cat([text_embeddings,info],dim=1)
            text_embed_gather = allgather(text_embeddings_with_info, n_gpu)
            vid_embed_gather = allgather(video_embedding_with_info, n_gpu)

            video_embeds_list.append(vid_embed_gather)
            text_embeds_list.append(text_embed_gather)


    text_embeds = torch.cat(text_embeds_list)
    vid_embeds = torch.cat(video_embeds_list).type_as(text_embeds)
    
    # sort by video ids
    _, v_indices = torch.sort(vid_embeds[:,-3], dim=0)
    vid_embeds_f = vid_embeds[v_indices,:]
    
    _, t_indices = torch.sort(text_embeds[:,-3], dim=0)
    text_embeds_f = text_embeds[t_indices,:]
    
    # gather the stats from all processes

    text_embeds_final = text_embeds_f[:,:768].view(-1, test_num_segment * test_num_crop, 768).mean(1)
    video_embeds_final = vid_embeds_f[:,:768].view(-1, test_num_segment * test_num_crop, 768).mean(1)
    metric_logger.synchronize_between_processes()
    dict_log = dict()
    if global_rank == 0:
        sims_tmp = sim_matrix(text_embeds_final, video_embeds_final).clone().detach().cpu().numpy()
    
        for metric in evl_metrics:
            metric_name = metric.__name__
            res = metric(sims_tmp)
            dict_log.update({metric_name:res})
            print(str(metric_name) + '  ' + str(res))

    return dict_log



