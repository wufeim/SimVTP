import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from modeling_finetune import Block, _cfg, PatchEmbed, get_sinusoid_encoding_table
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_ as __call_trunc_normal_
from transformers import AutoModelForMaskedLM
import timm
from engine_for_pretraining import sim_matrix
import numpy as np

def trunc_normal_(tensor, mean=0., std=1.):
    __call_trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)


__all__ = [
    'pretrain_simvtp_base_patch16_224', 
]


class PretrainVisionTransformerEncoder(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=0, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, init_values=None, tubelet_size=2,
                 use_learnable_pos_emb=False):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,tubelet_size=tubelet_size)
        num_patches = self.patch_embed.num_patches

        
        # TODO: Add the cls token
        if use_learnable_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        else:
            # sine-cosine positional embeddings 
            self.pos_embed = get_sinusoid_encoding_table(num_patches, embed_dim)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values)
            for i in range(depth)])


        self.norm =  norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        if use_learnable_pos_emb:
            trunc_normal_(self.pos_embed, std=.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x, mask, text_embedding, text_attention_mask, _mdm=False, vtc_vtm=False, video=False,text=False):
        x_vis = 0
        if video == True:
            _, _, T, _, _ = x.shape
            x = self.patch_embed(x)   # [bs, p_num, token_length]
            x = x + self.pos_embed.type_as(x).to(x.device).clone().detach()
            B, _, C = x.shape
            x_vis = x[~mask].reshape(B, -1, C) # ~mask means visible
            _, N_vis, C = x_vis.shape            
            video_attention_mask = torch.zeros([B,N_vis]).to(text_attention_mask.device)
            if vtc_vtm == True:
                x_vis = x_vis
                attention_mask = video_attention_mask[:,None,None,:]
        
        if text == True:
            text_attention_mask = (1 - text_attention_mask) * -10000
            if vtc_vtm == True:
                x_vis = text_embedding
                attention_mask = text_attention_mask[:,None,None,:]
            
        if _mdm == True:
            x_vis = torch.cat([x_vis,text_embedding],1)
            attention_mask = torch.cat([video_attention_mask,text_attention_mask],1)[:,None,None,:]

                    
        for blk in self.blocks:
            x_vis = blk(x_vis, attention_mask)

        x_vis = self.norm(x_vis)
        return x_vis

    def forward(self, x, mask, text_embedding, text_attention_mask, _mdm=False, vtc_vtm=False, video=False, text=False):
        x = self.forward_features(x, mask, text_embedding, text_attention_mask, _mdm, vtc_vtm, video, text)
        x = self.head(x)
        return x

class PretrainVisionTransformerDecoder(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, patch_size=16, num_classes=768, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, init_values=None, num_patches=196, tubelet_size=2
                 ):
        super().__init__()
        self.num_classes = num_classes
        assert num_classes == 3 * tubelet_size * patch_size ** 2 
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.patch_size = patch_size

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values)
            for i in range(depth)])
        self.norm =  norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward(self, x, return_token_num, attention_mask=None):
        if attention_mask == None:
            attention_mask = torch.zeros(x.shape[:2]).to(x.device)[:,None,None,:]
        for blk in self.blocks:
            x = blk(x, attention_mask)

        if return_token_num > 0:
            x = self.head(self.norm(x[:, -return_token_num:])) # only return the mask tokens predict pixels
        else:
            x = self.head(self.norm(x))

        return x

class PretrainVisionTransformer(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self,
                 img_size=224, 
                 patch_size=16, 
                 encoder_in_chans=3, 
                 encoder_num_classes=0, 
                 encoder_embed_dim=768, 
                 encoder_depth=12,
                 encoder_num_heads=12, 
                 decoder_num_classes=1536, #  decoder_num_classes=768, 
                 decoder_embed_dim=512, 
                 decoder_depth=8,
                 decoder_num_heads=8, 
                 mlp_ratio=4., 
                 qkv_bias=False, 
                 qk_scale=None, 
                 drop_rate=0., 
                 attn_drop_rate=0.,
                 drop_path_rate=0., 
                 norm_layer=nn.LayerNorm, 
                 init_values=0.,
                 use_learnable_pos_emb=False,
                 tubelet_size=2,
                 num_classes=0, # avoid the error from create_fn in timm
                 in_chans=0, # avoid the error from create_fn in timm
                 ):
        super().__init__()
        self.encoder = PretrainVisionTransformerEncoder(
            img_size=img_size, 
            patch_size=patch_size, 
            in_chans=encoder_in_chans, 
            num_classes=encoder_num_classes, 
            embed_dim=encoder_embed_dim, 
            depth=encoder_depth,
            num_heads=encoder_num_heads, 
            mlp_ratio=mlp_ratio, 
            qkv_bias=qkv_bias, 
            qk_scale=qk_scale, 
            drop_rate=drop_rate, 
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate, 
            norm_layer=norm_layer, 
            init_values=init_values,
            tubelet_size=tubelet_size,
            use_learnable_pos_emb=use_learnable_pos_emb)

        self.decoder = PretrainVisionTransformerDecoder(
            patch_size=patch_size, 
            num_patches=self.encoder.patch_embed.num_patches,
            num_classes=decoder_num_classes, 
            embed_dim=decoder_embed_dim, 
            depth=decoder_depth,
            num_heads=decoder_num_heads, 
            mlp_ratio=mlp_ratio, 
            qkv_bias=qkv_bias, 
            qk_scale=qk_scale, 
            drop_rate=drop_rate, 
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate, 
            norm_layer=norm_layer, 
            init_values=init_values,
            tubelet_size=tubelet_size)
        
        
        
        self.encoder_to_decoder = nn.Linear(encoder_embed_dim, decoder_embed_dim, bias=False)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.pos_embed = get_sinusoid_encoding_table(self.encoder.patch_embed.num_patches, decoder_embed_dim)

        trunc_normal_(self.mask_token, std=.02)

        self.text_embeddings = AutoModelForMaskedLM.from_pretrained('bert-base-uncased').bert.embeddings
        self.text_embeddings.train()
        print('text_embeddings has been initialized by bert weight.....')

        self.text_cls = nn.Linear(768,30522)
        self.vtm_head = nn.Linear(768, 2)  
        self._init_weights(self.text_cls)
        self.text_cls.train()

        self._init_weights(self.vtm_head)
        self.text_cls.train()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'mask_token'}

    def forward(self, x, mask, text_data):
        videos = x.clone()

        B, _, T, _, _ = x.shape   # [B, C, T, H, W]

        text_attention_mask = text_data['attention_mask']
        
        text_embedding = self.text_embeddings(input_ids = text_data['input_ids'])
        
        _, N_t, _ = text_embedding.shape

        x_vis_vt = self.encoder(x, mask, text_embedding, text_attention_mask, _mdm=True, vtc_vtm=False, video=True,text=True) # [B, N_vis, C_e]

        _ , N_vt, _ = x_vis_vt.shape

        #-mlm------------------------------------------------------------------------------------------------
        x_text = x_vis_vt[:, -N_t: ,: ]
        
        mlm_logits = self.text_cls(x_text)
        #-mvm-------------------------------------------------------------------------------------------------------
        x_vis_encoder_output = x_vis_vt[:,:N_vt - N_t,: ]
        
        x_vis = self.encoder_to_decoder(x_vis_encoder_output)
        
        C = x_vis.shape[2]
        expand_pos_embed = self.pos_embed.expand(B, -1, -1).type_as(x).to(x.device).clone().detach()
        
        pos_emd_vis = expand_pos_embed[~mask[:B,:]].reshape(B, -1, C)
        
        pos_emd_mask = expand_pos_embed[mask[:B,:]].reshape(B, -1, C)
        
        x_vis2decoder = torch.cat([x_vis + pos_emd_vis, self.mask_token + pos_emd_mask], dim=1) # [B, N, C_d] 
        
        x_full = self.decoder(x_vis2decoder, pos_emd_mask.shape[1]) # [B, N_mask, 3 * 16 * 16]
        #-vtc------------------------------------------------------------------------------------------------
        x_video_single = self.encoder(x, mask, text_embedding, text_attention_mask, _mdm=False, vtc_vtm=True, video=True,text=False)
        
        x_text_single = self.encoder(x, mask, text_embedding, text_attention_mask, _mdm=False, vtc_vtm=True, video=False,text=True)
        #-vtm------------------------------------------------------------------------------------------------
        # Random sample
        vtm_flag = True
        if vtm_flag:
            pos_len = B
            neg_len = pos_len
            neg_video_list = []
            neg_video_mask_list =[]
            for i in range(pos_len):
                idx = generate_random(0, pos_len, flag=i)              
                neg_video_list.append(videos[idx])
                neg_video_mask_list.append(mask[idx])

        video_neg = torch.stack(neg_video_list,dim=0)  
         
        mask_neg = torch.stack(neg_video_mask_list,dim=0)   

        text_all = torch.cat([text_data['mlm_input_ids'], text_data['mlm_input_ids']],dim=0)     
        
        text_atts_all = torch.cat([text_data['attention_mask'], text_data['attention_mask']],dim=0)    
         
        text_token_vtm = {'input_ids': text_all,'attention_mask':  text_atts_all}

        videos_all_vtm = torch.cat([videos,video_neg],dim=0)
        
        bool_masked_pos_vtm = torch.cat([mask,mask_neg],dim=0)

        text_attention_mask_vtm = text_token_vtm['attention_mask']

        text_embedding_vtm = self.text_embeddings(input_ids = text_token_vtm['input_ids'])
        
        vtm_feature = self.encoder(videos_all_vtm, bool_masked_pos_vtm, text_embedding_vtm, text_attention_mask_vtm, _mdm=True, vtc_vtm=False, video=True,text=True) # [B, N_vis, C_e]
        
        vtm_logits = self.vtm_head(vtm_feature.mean(1))

        vtm_labels = torch.cat([torch.ones(B,dtype=torch.long),torch.zeros(B,dtype=torch.long)], dim=0).to(videos.device)


        return  x_full, mlm_logits ,vtm_logits, vtm_labels, x_text_single.mean(1), x_video_single.mean(1)
        

def generate_random(start, end, flag):
    """
    idx != flag
    """
    idx = 0
    while(True):
        idx = torch.randint(start,end,(1,)).item()
        if idx != flag:
            return idx

@register_model
def pretrain_simvtp_base_patch16_224(pretrained=False, **kwargs):
    model = PretrainVisionTransformer(
        img_size=224,
        patch_size=16, 
        encoder_embed_dim=768, 
        encoder_depth=12, 
        encoder_num_heads=12,
        encoder_num_classes=0,
        decoder_num_classes=1536,
        decoder_embed_dim=384,
        decoder_num_heads=6,
        mlp_ratio=4, 
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), 
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model
 