from functools import partial
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

import timm
import loralib as lora

from models.pos_embed import interpolate_pos_embed


# -------------------------------------------------------------------------
# Paths — update these to match your environment
# -------------------------------------------------------------------------
# WEIGHT_DIR      : pretrained backbone weights, one state_dict per backbone saved as <arch>.pth.tar
#                   (retfound, retfound_dinov2, mae, openclip, dinov2, dinov3, dinov3_vith16plus)
# DINOV3_REPO     : local clone of https://github.com/facebookresearch/dinov3 (loaded with torch.hub, source='local')
# DINOV3_L_WEIGHTS: official DINOv3 ViT-L/16 LVD-1689M checkpoint (the hub reads the hash in the file name)
# -------------------------------------------------------------------------
WEIGHT_DIR       = Path('/path/to/weights/')
DINOV3_REPO      = '/path/to/dinov3'
DINOV3_L_WEIGHTS = WEIGHT_DIR / 'dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth'


class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """
    Vision Transformer used for RETFound-MAE (returns the normalized token sequence from forward_features)
    """
    def __init__(self, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)

    '''
    modified from https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py
    '''
    def forward_features(self, x, attn_mask=None, is_causal=False):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)

        return x

def RETFound_mae(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def unfreeze(module):
    for p in module.parameters():
        p.requires_grad = True

def freeze(module):
    for p in module.parameters():
        p.requires_grad = False

def _is_full_ft(ft_blks):
    return isinstance(ft_blks, str) and ft_blks.lower() == "full"

def build_model(args):
    '''
    ViT backbone + single-logit head for dementia detection.

    args requires:
    - arch       # retfound | retfound_dinov2 | mae | openclip | dinov2 | dinov3 | dinov3_vith16plus
    - img_size
    - ft         # linear | partial | lora
    - ft_blks    # int (number of final blocks) or string "full"
    - lora_rank
    '''
    fn = f'{args.arch}.pth.tar'
    ckpt = torch.load(WEIGHT_DIR/fn, map_location='cpu')

    if args.arch == 'dinov2':
        model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14', pretrained=False)
    elif args.arch == 'dinov3':
        # DINOv3 ViT-L/16 (300M, embed_dim=1024); rotary position embeddings, so the input size needs no argument
        model = torch.hub.load(DINOV3_REPO, 'dinov3_vitl16', source='local',
                               weights=str(DINOV3_L_WEIGHTS))
    elif args.arch == 'dinov3_vith16plus':
        # DINOv3 ViT-H+/16 (840M, embed_dim=1280); the official state_dict saved as dinov3_vith16plus.pth.tar
        model = torch.hub.load(DINOV3_REPO, 'dinov3_vith16plus', source='local',
                               weights=str(WEIGHT_DIR/fn))
    elif args.arch == 'openclip':
        model = timm.create_model('vit_large_patch14_clip_224.laion2b', pretrained=False, num_classes=0,
                                  img_size=args.img_size, dynamic_img_size=True)
    elif args.arch == 'mae':
        model = timm.create_model('vit_large_patch16_224.mae', pretrained=False, num_classes=0,
                                  img_size=args.img_size, dynamic_img_size=True)
    elif args.arch == 'retfound':
        model = RETFound_mae(img_size=args.img_size, num_classes=0, global_pool='token')
        interpolate_pos_embed(model, ckpt)
    elif args.arch == 'retfound_dinov2':
        model = timm.create_model('vit_large_patch14_dinov2.lvd142m', pretrained=False,
                                  img_size=args.img_size, num_classes=0)
    else:
        raise ValueError(f"Unknown arch '{args.arch}'")

    # without LoRA, load the pretrained weights as they are
    if args.ft != 'lora':
        model.load_state_dict(ckpt)

    # freeze the whole backbone first
    freeze(model)

    if args.ft == 'lora':
        unfreeze(model.norm)

        # target blocks: all blocks for "full", otherwise the last N blocks
        blocks = list(model.blocks)
        if _is_full_ft(args.ft_blks):
            target_blocks = blocks
        else:
            n = int(args.ft_blks)
            target_blocks = list(reversed(blocks))[:n]

        # replace qkv of each target block with a LoRA MergedLinear (LoRA on query and value)
        for blk in target_blocks:
            if hasattr(blk, "attn") and hasattr(blk.attn, "qkv"):
                in_f = blk.attn.qkv.in_features
                out_f = blk.attn.qkv.out_features
                blk.attn.qkv = lora.MergedLinear(
                    in_f, out_f,
                    r=args.lora_rank,
                    lora_alpha=args.lora_rank,
                    enable_lora=[True, False, True],
                    merge_weights=False
                )

        # re-apply the pretrained weights (strict=False because of the new LoRA parameters)
        model.load_state_dict(ckpt, strict=False)

        # train only the LoRA matrices (this also re-freezes the final norm); the new head is added below
        lora.mark_only_lora_as_trainable(model)

    elif args.ft == 'partial':
        # final normalization layer + last ft_blks transformer blocks
        unfreeze(model.norm)
        for i, blk in enumerate(reversed(list(model.blocks))):
            if i >= args.ft_blks:
                break
            unfreeze(blk)

    # new trainable classification head on the class token (embed_dim read from the backbone: 1024 or 1280)
    embed_dim = getattr(model, 'embed_dim', 1024)
    model.head = nn.Linear(embed_dim, 1)
    return model
