# -*- coding: utf-8 -*-
"""
Single-script loader:
- build DepthAnythingV2 DINOv2 encoder (DinoVisionTransformer + DINOv2())
- load encoder weights from a DepthAnythingV2 checkpoint
- extract 12-layer patch tokens (and optional cls tokens)

Place this file inside your Depth-Anything-V2 codebase so that:
  from dinov2_layers import Mlp, PatchEmbed, SwiGLUFFNFused, MemEffAttention, NestedTensorBlock
works.
"""

from functools import partial
import math
import logging
from typing import Sequence, Tuple, Union, Callable
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_

# ====== IMPORTANT: these come from Depth-Anything-V2 repo ======
# make sure this script is in the same package path, or adjust the import
from depth_anything_v2.dinov2_layers import Mlp, PatchEmbed, SwiGLUFFNFused, MemEffAttention, NestedTensorBlock as Block


logger = logging.getLogger("dinov2")


# -----------------------------
# 1) DINOv2 Encoder (your pasted code)
# -----------------------------
def named_apply(fn: Callable, module: nn.Module, name="", depth_first=True, include_root=False) -> nn.Module:
    if not depth_first and include_root:
        fn(module=module, name=name)
    for child_name, child_module in module.named_children():
        child_name = ".".join((name, child_name)) if name else child_name
        named_apply(fn=fn, module=child_module, name=child_name, depth_first=depth_first, include_root=True)
    if depth_first and include_root:
        fn(module=module, name=name)
    return module


class BlockChunk(nn.ModuleList):
    def forward(self, x):
        for b in self:
            x = b(x)
        return x


class DinoVisionTransformer(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        ffn_bias=True,
        proj_bias=True,
        drop_path_rate=0.0,
        drop_path_uniform=False,
        init_values=None,
        embed_layer=PatchEmbed,
        act_layer=nn.GELU,
        block_fn=Block,
        ffn_layer="mlp",
        block_chunks=1,
        num_register_tokens=0,
        interpolate_antialias=False,
        interpolate_offset=0.1,
    ):
        super().__init__()
        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.num_features = self.embed_dim = embed_dim
        self.num_tokens = 1
        self.n_blocks = depth
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.num_register_tokens = num_register_tokens
        self.interpolate_antialias = interpolate_antialias
        self.interpolate_offset = interpolate_offset

        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        assert num_register_tokens >= 0
        self.register_tokens = nn.Parameter(torch.zeros(1, num_register_tokens, embed_dim)) if num_register_tokens else None

        if drop_path_uniform is True:
            dpr = [drop_path_rate] * depth
        else:
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        if ffn_layer == "mlp":
            ffn_layer = Mlp
        elif ffn_layer in ("swiglufused", "swiglu"):
            ffn_layer = SwiGLUFFNFused
        elif ffn_layer == "identity":
            def f(*args, **kwargs):
                return nn.Identity()
            ffn_layer = f
        else:
            raise NotImplementedError(f"Unknown ffn_layer={ffn_layer}")

        blocks_list = [
            block_fn(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
                ffn_bias=ffn_bias,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
                ffn_layer=ffn_layer,
                init_values=init_values,
            )
            for i in range(depth)
        ]

        if block_chunks > 0:
            self.chunked_blocks = True
            chunked_blocks = []
            chunksize = depth // block_chunks
            for i in range(0, depth, chunksize):
                chunked_blocks.append([nn.Identity()] * i + blocks_list[i : i + chunksize])
            self.blocks = nn.ModuleList([BlockChunk(p) for p in chunked_blocks])
        else:
            self.chunked_blocks = False
            self.blocks = nn.ModuleList(blocks_list)

        self.norm = norm_layer(embed_dim)
        self.head = nn.Identity()
        self.mask_token = nn.Parameter(torch.zeros(1, embed_dim))

        self.init_weights()

    def init_weights(self):
        trunc_normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.cls_token, std=1e-6)
        if self.register_tokens is not None:
            nn.init.normal_(self.register_tokens, std=1e-6)
        named_apply(init_weights_vit_timm, self)

    def interpolate_pos_encoding(self, x, w, h):
        previous_dtype = x.dtype
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        pos_embed = self.pos_embed.float()
        class_pos_embed = pos_embed[:, 0]
        patch_pos_embed = pos_embed[:, 1:]
        dim = x.shape[-1]

        w0 = w // self.patch_size
        h0 = h // self.patch_size
        w0, h0 = w0 + self.interpolate_offset, h0 + self.interpolate_offset

        sqrt_N = math.sqrt(N)
        sx, sy = float(w0) / sqrt_N, float(h0) / sqrt_N
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(sqrt_N), int(sqrt_N), dim).permute(0, 3, 1, 2),
            scale_factor=(sx, sy),
            mode="bicubic",
            antialias=self.interpolate_antialias,
        )
        assert int(w0) == patch_pos_embed.shape[-2]
        assert int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1).to(previous_dtype)

    def prepare_tokens_with_masks(self, x, masks=None):
        B, nc, w, h = x.shape
        x = self.patch_embed(x)
        if masks is not None:
            x = torch.where(masks.unsqueeze(-1), self.mask_token.to(x.dtype).unsqueeze(0), x)

        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = x + self.interpolate_pos_encoding(x, w, h)

        if self.register_tokens is not None:
            x = torch.cat((x[:, :1], self.register_tokens.expand(x.shape[0], -1, -1), x[:, 1:]), dim=1)
        return x

    def _get_intermediate_layers_not_chunked(self, x, n=1):
        x = self.prepare_tokens_with_masks(x)
        output, total_block_len = [], len(self.blocks)
        blocks_to_take = range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in blocks_to_take:
                output.append(x)
        assert len(output) == len(blocks_to_take), f"only {len(output)} / {len(blocks_to_take)} blocks found"
        return output

    def _get_intermediate_layers_chunked(self, x, n=1):
        x = self.prepare_tokens_with_masks(x)
        output, i, total_block_len = [], 0, len(self.blocks[-1])
        blocks_to_take = range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        for block_chunk in self.blocks:
            for blk in block_chunk[i:]:
                x = blk(x)
                if i in blocks_to_take:
                    output.append(x)
                i += 1
        assert len(output) == len(blocks_to_take), f"only {len(output)} / {len(blocks_to_take)} blocks found"
        return output

    def get_intermediate_layers(
        self,
        x: torch.Tensor,
        n: Union[int, Sequence] = 1,
        reshape: bool = False,
        return_class_token: bool = False,
        norm: bool = True,
    ):
        if self.chunked_blocks:
            outputs = self._get_intermediate_layers_chunked(x, n)
        else:
            outputs = self._get_intermediate_layers_not_chunked(x, n)

        if norm:
            outputs = [self.norm(out) for out in outputs]

        class_tokens = [out[:, 0] for out in outputs]
        outputs = [out[:, 1 + self.num_register_tokens :] for out in outputs]  # drop cls + reg

        if reshape:
            B, _, w, h = x.shape
            outputs = [
                out.reshape(B, w // self.patch_size, h // self.patch_size, -1)
                   .permute(0, 3, 1, 2).contiguous()
                for out in outputs
            ]
        if return_class_token:
            return tuple(zip(outputs, class_tokens))
        return tuple(outputs)


def init_weights_vit_timm(module: nn.Module, name: str = ""):
    if isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def vit_small(patch_size=16, num_register_tokens=0, **kwargs):
    return DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        block_fn=partial(Block, attn_class=MemEffAttention),
        num_register_tokens=num_register_tokens,
        **kwargs,
    )


def vit_base(patch_size=16, num_register_tokens=0, **kwargs):
    return DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        block_fn=partial(Block, attn_class=MemEffAttention),
        num_register_tokens=num_register_tokens,
        **kwargs,
    )


def vit_large(patch_size=16, num_register_tokens=0, **kwargs):
    return DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        block_fn=partial(Block, attn_class=MemEffAttention),
        num_register_tokens=num_register_tokens,
        **kwargs,
    )


def vit_giant2(patch_size=16, num_register_tokens=0, **kwargs):
    return DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=1536,
        depth=40,
        num_heads=24,
        mlp_ratio=4,
        block_fn=partial(Block, attn_class=MemEffAttention),
        num_register_tokens=num_register_tokens,
        **kwargs,
    )


def DINOv2(model_name: str):
    model_zoo = {"vits": vit_small, "vitb": vit_base, "vitl": vit_large, "vitg": vit_giant2}
    return model_zoo[model_name](
        img_size=518,
        patch_size=14,
        init_values=1.0,
        ffn_layer="mlp" if model_name != "vitg" else "swiglufused",
        block_chunks=0,
        num_register_tokens=0,
        interpolate_antialias=False,
        interpolate_offset=0.1,
    )


# -----------------------------
# 2) Load DepthAnythingV2 encoder(pretrained) weights into this DINOv2
# -----------------------------
def _strip_module_prefix(sd: dict) -> dict:
    return {k.replace("module.", ""): v for k, v in sd.items()}


def extract_encoder_sd_from_depthanything_ckpt(ckpt: dict) -> dict:
    """
    Return encoder(pretrained) state_dict with keys matching DinoVisionTransformer.
    """
    sd = ckpt.get("state_dict", ckpt.get("model", ckpt))
    if not isinstance(sd, dict):
        raise RuntimeError("ckpt does not contain a dict state_dict / model")
    sd = _strip_module_prefix(sd)

    candidate_prefixes = [
        "pretrained.",
        "model.pretrained.",
        "backbone.pretrained.",
        "net.pretrained.",
        "encoder.pretrained.",
        "student.pretrained.",
    ]

    enc_sd = {}
    for k, v in sd.items():
        for p in candidate_prefixes:
            if k.startswith(p):
                enc_sd[k[len(p) :]] = v
                break

    if len(enc_sd) > 0:
        return enc_sd

    # fallback: any key containing ".pretrained."
    for k, v in sd.items():
        if ".pretrained." in k:
            enc_sd[k.split(".pretrained.", 1)[1]] = v

    if len(enc_sd) == 0:
        sample = list(sd.keys())[:40]
        raise RuntimeError(
            "Cannot find encoder(pretrained) weights in ckpt.\n"
            f"Sample ckpt keys: {sample}"
        )
    return enc_sd


def load_depthanything_encoder_weights(dino: nn.Module, ckpt_path: str, device: str):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    enc_sd = extract_encoder_sd_from_depthanything_ckpt(ckpt)

    missing, unexpected = dino.load_state_dict(enc_sd, strict=False)
    print(f"[LOAD] encoder keys={len(enc_sd)} from {ckpt_path}")
    print(f"       missing={len(missing)} unexpected={len(unexpected)}")
    # 如需细看：
    # print("missing:", missing)
    # print("unexpected:", unexpected)

    dino.to(device).eval()
    return dino


# -----------------------------
# 3) Extract 12 layer features
# -----------------------------
@torch.inference_mode()
def extract_12_layers(dino: DinoVisionTransformer, x: torch.Tensor, with_cls: bool = False):
    n_blocks = len(dino.blocks[-1]) if getattr(dino, "chunked_blocks", False) else len(dino.blocks)
    if n_blocks < 12:
        raise RuntimeError(f"encoder has only {n_blocks} blocks, cannot extract 12 layers")

    if with_cls:
        feats = dino.get_intermediate_layers(
            x, n=list(range(12)), reshape=False, return_class_token=True, norm=True
        )
        # feats: tuple of (patch, cls)
        return feats
    else:
        feats = dino.get_intermediate_layers(
            x, n=list(range(12)), reshape=False, return_class_token=False, norm=True
        )
        # feats: tuple of patch tokens
        return feats


# def main():
#     # ======= YOU CHANGE THESE =======
#     CKPT_PATH = r"E:\code\Depth-Anything-V2-main\checkpoints\depth_anything_v2_vitb.pth"
#     ENCODER_NAME = "vitb"  # vits/vitb/vitl/vitg
#     DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
#
#     assert Path(CKPT_PATH).exists(), f"ckpt not found: {CKPT_PATH}"
#
#     # build encoder
#     dino = DINOv2(ENCODER_NAME)
#
#     # load weights
#     dino = load_depthanything_encoder_weights(dino, CKPT_PATH, device=DEVICE)
#
#     # input (H,W should be multiple of 14)
#     x = torch.randn(1, 3, 640, 480, device=DEVICE)
#
#     # extract 12 layer patch tokens
#     layers = extract_12_layers(dino, x, with_cls=False)
#     print("num layers:", len(layers))
#     for i, t in enumerate(layers):
#         print(f"layer[{i:02d}] {tuple(t.shape)}  mean={t.mean().item():.4f}  std={t.std().item():.4f}")
#
#
#
# if __name__ == "__main__":
#     main()
