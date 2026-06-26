

import os
import glob
from typing import Tuple, Union, Optional

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F

from fusion2 import IRVISFusionNet


# -------------------------

# -------------------------
def _normalize_path(p: Union[str, os.PathLike, tuple, list, None]) -> str:
    if p is None:
        return ""
    if isinstance(p, (tuple, list)):
        if len(p) == 0:
            return ""
        p = p[0]
    if isinstance(p, os.PathLike):
        p = os.fspath(p)
    if not isinstance(p, str):
        return ""
    return p


from collections import Counter

def load_ckpt_if_exists(model, ckpt_path, device="cpu", strict=False):
    ckpt = torch.load(ckpt_path, map_location=device)
    sd = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt

    res = model.load_state_dict(sd, strict=strict)

    missing = res.missing_keys
    unexpected = res.unexpected_keys

    print(f"[LOAD] {ckpt_path}")
    print(f"       missing={len(missing)} unexpected={len(unexpected)} strict={strict}")


    pref2 = [".".join(k.split(".")[:2]) for k in missing]
    c = Counter(pref2)
    print("[MISSING prefix2 top20]")
    for k, v in c.most_common(20):
        print(f"  {k}: {v}")


    print("[MISSING sample]")
    for k in missing[:30]:
        print(" ", k)

    return ckpt



# -------------------------
# -------------------------
def pil_to_gray_tensor(img: Image.Image, hw: Tuple[int, int]) -> torch.Tensor:

    g = img.convert("L").resize((hw[1], hw[0]), resample=Image.BILINEAR)
    arr = np.array(g, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


def pil_to_rgb_uint8(img: Image.Image, hw: Tuple[int, int]) -> np.ndarray:

    rgb = img.convert("RGB").resize((hw[1], hw[0]), resample=Image.BILINEAR)
    return np.array(rgb, dtype=np.uint8)


def ycbcr_to_rgb_uint8(Y: np.ndarray, Cb: np.ndarray, Cr: np.ndarray) -> np.ndarray:
  
    ycbcr = np.stack([Y, Cb, Cr], axis=-1).astype(np.uint8)
    rgb = Image.fromarray(ycbcr, mode="YCbCr").convert("RGB")
    return np.array(rgb, dtype=np.uint8)


# -------------------------

# -------------------------
@torch.no_grad()
def run_test(
    ir_dir: str,
    vi_dir: str,
    out_dir: str,
    fusion_ckpt_path: str,
    depth_ckpt_path: str,
    hw: Tuple[int, int] = (480, 640),
    device: str = "cuda",


    dino_model_name: str = "convnext_base.dinov3_lvd1689m",
    depth_pool_Ms: Tuple[int, int, int, int] = (0, 0, 0, 0),
    depth_layer_ids_4: Tuple[int, int, int, int] = (2, 5, 8, 11),
    hg_layers: int = 1,
    hg_heads: int = 8,
    hg_d_model: int = 256,
    decoder_embed_dim: int = 64,
    decoder_fuse_to: str = "middle",
):
    os.makedirs(out_dir, exist_ok=True)

    ir_paths = sorted(glob.glob(os.path.join(ir_dir, "*")))
    vi_paths = sorted(glob.glob(os.path.join(vi_dir, "*")))
    assert len(ir_paths) == len(vi_paths), "IR/VI count mismatch"


    fusion = IRVISFusionNet(
        depth_ckpt_path=depth_ckpt_path,
        train_hw=hw,
        dino_model_name=dino_model_name,
        depth_pool_Ms=depth_pool_Ms,
        depth_layer_ids_4=depth_layer_ids_4,
        hg_layers=hg_layers,
        hg_heads=hg_heads,
        hg_d_model=hg_d_model,
        decoder_embed_dim=decoder_embed_dim,
        decoder_fuse_to=decoder_fuse_to,
    ).to(device)
    fusion.eval()


    load_ckpt_if_exists(fusion, fusion_ckpt_path, device=device, strict=False)


    for ip, vp in zip(ir_paths, vi_paths):
        name = os.path.splitext(os.path.basename(ip))[0]

        ir_img = Image.open(ip)
        vi_img = Image.open(vp)



        ir_t = pil_to_gray_tensor(ir_img, hw).unsqueeze(0).to(device)
        vi_gray_t = pil_to_gray_tensor(vi_img, hw).unsqueeze(0).to(device)


        fused = fusion(ir_t, vi_gray_t)
        fused = fused.clamp(0.0, 1.0)


        Y = (fused[0, 0].detach().cpu().numpy() * 255.0 + 0.5).astype(np.uint8)


        vi_rgb_u8 = pil_to_rgb_uint8(vi_img, hw)
        vi_ycbcr = Image.fromarray(vi_rgb_u8, mode="RGB").convert("YCbCr")
        vi_ycbcr_u8 = np.array(vi_ycbcr, dtype=np.uint8)
        Cb = vi_ycbcr_u8[..., 1]
        Cr = vi_ycbcr_u8[..., 2]


        out_rgb = ycbcr_to_rgb_uint8(Y, Cb, Cr)


        out_path = os.path.join(out_dir, f"{name}.png")
        Image.fromarray(out_rgb).save(out_path)

        print(f"[OK] {name} -> {out_path}")


if __name__ == "__main__":
    run_test(
        ir_dir="",
        vi_dir="",
        out_dir="",


        fusion_ckpt_path="fusion_best.pth",


        depth_ckpt_path="depth_anything_v2_vitb.pth",

        hw=(600, 800),
        device="cuda",


        dino_model_name="convnext_base.dinov3_lvd1689m",
        depth_pool_Ms=(0, 0, 0, 0),
        depth_layer_ids_4=(2, 5, 8, 11),
        hg_layers=1,
        hg_heads=8,
        hg_d_model=256,
        decoder_embed_dim=64,
        decoder_fuse_to="middle",
    )
