
import os
import glob
from typing import Tuple, Optional, Union

import numpy as np
from PIL import Image

from PIDNet import PIDNet
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from fusion2 import IRVISFusionNet, FusionPIDNetSystem


# ============================================================

# ============================================================

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


def filtered_state_dict(model: torch.nn.Module, exclude_prefixes=()):

    sd = model.state_dict()
    if exclude_prefixes:
        sd = {k: v for k, v in sd.items() if not any(k.startswith(pref) for pref in exclude_prefixes)}
    return sd


def load_ckpt_if_exists(
    model: torch.nn.Module,
    ckpt_path: Union[str, os.PathLike, tuple, list, None],
    device: str = "cpu",
    strict: bool = False
) -> int:

    ckpt_path = _normalize_path(ckpt_path)
    if ckpt_path == "" or (not os.path.isfile(ckpt_path)):
        print(f"[LOAD] skip (not found): {ckpt_path}")
        return 0

    ckpt = torch.load(ckpt_path, map_location=device)
    sd = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt

    res = model.load_state_dict(sd, strict=strict)

    if isinstance(res, tuple):
        missing, unexpected = res
    else:
        missing, unexpected = res.missing_keys, res.unexpected_keys

    print(f"[LOAD] {ckpt_path}")
    print(f"       missing={len(missing)} unexpected={len(unexpected)} strict={strict}")

    if isinstance(ckpt, dict) and "epoch" in ckpt:
        return int(ckpt["epoch"]) + 1
    return 0


# ============================================================
# ============================================================

class IRVISSegDataset(Dataset):
    def __init__(self, ir_dir: str, vi_dir: str, mask_dir: str, hw: Tuple[int, int] = (448, 448)):
        self.ir_paths = sorted(glob.glob(os.path.join(ir_dir, "*")))
        self.vi_paths = sorted(glob.glob(os.path.join(vi_dir, "*")))
        assert len(self.ir_paths) == len(self.vi_paths), "IR/VI count mismatch"

        self.mask_paths = [os.path.join(mask_dir, os.path.basename(p)) for p in self.ir_paths]
        self.hw = hw

    def __len__(self):
        return len(self.ir_paths)

    def _read_gray(self, path: str) -> torch.Tensor:
        img = Image.open(path).convert("L")
        img = img.resize((self.hw[1], self.hw[0]), resample=Image.BILINEAR)
        arr = np.array(img, dtype=np.float32) / 255.0
        return torch.from_numpy(arr).unsqueeze(0)  # (1,H,W)

    def _read_mask(self, path: str) -> torch.Tensor:
        m = Image.open(path)
        m = m.resize((self.hw[1], self.hw[0]), resample=Image.NEAREST)
        arr = np.array(m, dtype=np.int64)
        return torch.from_numpy(arr)  # (H,W)

    def __getitem__(self, idx):
        ir = self._read_gray(self.ir_paths[idx])
        vi = self._read_gray(self.vi_paths[idx])
        mask = self._read_mask(self.mask_paths[idx])
        return ir, vi, mask


# ============================================================
# ============================================================

def l1_loss_fusion(fused: torch.Tensor, ir: torch.Tensor, vi: torch.Tensor, w_vi: float = 1.0, w_ir: float = 1.2):
    return w_vi * F.l1_loss(fused, vi) + w_ir * F.l1_loss(fused, ir)

def l1_loss_fusion_max(fused: torch.Tensor, ir: torch.Tensor, vi: torch.Tensor, w: float = 1.0):

    target = torch.maximum(ir, vi)
    return w * F.l1_loss(fused, target)

class SobelGrad(nn.Module):
    def __init__(self):
        super().__init__()
        kx = torch.tensor([[-1, 0, 1],
                           [-2, 0, 2],
                           [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        ky = torch.tensor([[-1, -2, -1],
                           [0, 0, 0],
                           [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer("kx", kx)
        self.register_buffer("ky", ky)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gx = F.conv2d(x, self.kx, padding=1)
        gy = F.conv2d(x, self.ky, padding=1)
        return torch.abs(gx) + torch.abs(gy)


def grad_loss_fusion(fused: torch.Tensor, ir: torch.Tensor, vi: torch.Tensor, sobel: SobelGrad):
    gf = sobel(fused)
    gir = sobel(ir)
    gvi = sobel(vi)
    target = torch.max(gir, gvi)
    return F.l1_loss(gf, target)


def mask_to_boundary(mask: torch.Tensor, ignore_index: int = 255, dilate: int = 1) -> torch.Tensor:

    B, H, W = mask.shape
    valid = (mask != ignore_index)

    edge = torch.zeros((B, H, W), device=mask.device, dtype=torch.bool)

    diff_h = (mask[:, :, 1:] != mask[:, :, :-1]) & valid[:, :, 1:] & valid[:, :, :-1]
    edge[:, :, 1:] |= diff_h
    edge[:, :, :-1] |= diff_h

    diff_v = (mask[:, 1:, :] != mask[:, :-1, :]) & valid[:, 1:, :] & valid[:, :-1, :]
    edge[:, 1:, :] |= diff_v
    edge[:, :-1, :] |= diff_v

    edge = edge.float().unsqueeze(1)

    if dilate and dilate > 0:
        edge = F.max_pool2d(edge, kernel_size=2 * dilate + 1, stride=1, padding=dilate)

    return edge


def pidnet_seg_loss(
    pid_out,
    mask: torch.Tensor,
    ignore_index: int = 255,
    aux_weight: float = 0.4,
    use_detail_loss: bool = False,
    detail_weight: float = 0.1,
):

    if isinstance(pid_out, (list, tuple)):
        x_extra_p, x_main, x_extra_d = pid_out
    else:
        x_extra_p, x_main, x_extra_d = None, pid_out, None

    if x_main.shape[-2:] != mask.shape[-2:]:
        x_main_up = F.interpolate(x_main, size=mask.shape[-2:], mode="bilinear", align_corners=False)
    else:
        x_main_up = x_main

    loss = F.cross_entropy(x_main_up, mask.long(), ignore_index=ignore_index)

    if x_extra_p is not None:
        x_aux = F.interpolate(x_extra_p, size=mask.shape[-2:], mode="bilinear", align_corners=False)
        loss = loss + aux_weight * F.cross_entropy(x_aux, mask.long(), ignore_index=ignore_index)

    if use_detail_loss and (x_extra_d is not None):
        x_d = F.interpolate(x_extra_d, size=mask.shape[-2:], mode="bilinear", align_corners=False)
        bd = mask_to_boundary(mask, ignore_index=ignore_index, dilate=1)
        loss = loss + detail_weight * F.binary_cross_entropy_with_logits(x_d, bd)

    return loss


# ============================================================
# ============================================================

def train(
    ir_dir: str,
    vi_dir: str,
    mask_dir: str,
    depth_ckpt_path: str,
    num_classes: int,
    hw: Tuple[int, int] = (448, 448),
    batch_size: int = 2,
    epochs: int = 50,
    lr_fusion: float = 2e-4,
    lr_seg: float = 1e-3,
    weight_decay: float = 1e-2,
    device: str = "cuda",

    lambda_l1: float = 1.0,
    lambda_grad: float = 100.0,
    lambda_seg: float = 1.0,

    seg_warmup_epochs: int = 5,
    ignore_index: int = 255,

    pidnet_augment: bool = True,
    use_detail_loss: bool = False,

    dino_model_name: str = "convnext_base.dinov3_lvd1689m",
    depth_pool_Ms: Tuple[int, int, int, int] = (0, 0, 0, 0),


    fusion_resume_path: str = "",
    pidnet_resume_path: str = "",
):
    ds = IRVISSegDataset(ir_dir, vi_dir, mask_dir, hw=hw)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)

    fusion = IRVISFusionNet(
        depth_ckpt_path=depth_ckpt_path,
        train_hw=hw,
        dino_model_name=dino_model_name,
        depth_pool_Ms=depth_pool_Ms,
        depth_layer_ids_4=(2, 5, 8, 11),
        hg_layers=1,
        hg_heads=8,
        hg_d_model=256,
        decoder_embed_dim=64,
        decoder_fuse_to="middle",
    ).to(device)

    pidnet = PIDNet(num_classes=num_classes, augment=pidnet_augment).to(device)
    system = FusionPIDNetSystem(fusion, pidnet).to(device)


    start_ep_f = load_ckpt_if_exists(system.fusion, fusion_resume_path, device=device, strict=False)
    start_ep_p = load_ckpt_if_exists(system.pidnet, pidnet_resume_path, device=device, strict=False)
    start_epoch = max(start_ep_f, start_ep_p, 0)
    if start_epoch > 0:
        print(f"[RESUME] start_epoch={start_epoch} (max of fusion/pid epochs)")

    def count_trainable(m):
        return sum(p.numel() for p in m.parameters() if p.requires_grad)

    print("trainable fusion:", count_trainable(system.fusion))
    print("trainable pidnet:", count_trainable(system.pidnet))


    sobel = SobelGrad().to(device)


    fusion_params = [p for p in system.fusion.parameters() if p.requires_grad]
    pidnet_params = [p for p in system.pidnet.parameters() if p.requires_grad]

    opt = torch.optim.AdamW(
        [
            {"params": fusion_params, "lr": lr_fusion},
            {"params": pidnet_params, "lr": lr_seg},
        ],
        weight_decay=weight_decay
    )

    scaler = torch.cuda.amp.GradScaler(enabled=device.startswith("cuda"))

    for ep in range(start_epoch, epochs):
        system.train()

        for it, (ir, vi, mask) in enumerate(dl):
            ir = ir.to(device, non_blocking=True)
            vi = vi.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=device.startswith("cuda")):
                fused, pid_out = system(ir, vi)

                l_l1 = l1_loss_fusion(fused, ir, vi,w_vi=1.0,w_ir=1.2)
                l_g = grad_loss_fusion(fused, ir, vi, sobel)

                if ep < seg_warmup_epochs:
                    l_s = torch.tensor(0.0, device=device)
                    seg_w = 0.0
                else:
                    l_s = pidnet_seg_loss(
                        pid_out, mask,
                        ignore_index=ignore_index,
                        aux_weight=0.4,
                        use_detail_loss=use_detail_loss,
                        detail_weight=0.1
                    )
                    seg_w = lambda_seg

                loss = lambda_l1 * l_l1 + lambda_grad * l_g + seg_w * l_s

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            if it % 20 == 0:
                print(
                    f"[ep {ep:03d} it {it:05d}] "
                    f"loss={loss.item():.4f} l1={l_l1.item():.4f} grad={l_g.item():.4f} seg={l_s.item():.4f}"
                )


        os.makedirs("checkpoints", exist_ok=True)


        fusion_sd = filtered_state_dict(
            system.fusion,
            exclude_prefixes=("backbone.dino.", "backbone.depth.")
        )

        torch.save(
            {
                "state_dict": fusion_sd,
                "epoch": ep,
                "dino_model_name": dino_model_name,
                "depth_pool_Ms": depth_pool_Ms,
                "hw": hw,
            },
            f"checkpoints_fmb/fusion_ep{ep:03d}.pth"
        )


        pidnet_sd = system.pidnet.state_dict()
        torch.save(
            {
                "state_dict": pidnet_sd,
                "epoch": ep,
                "num_classes": num_classes,
                "hw": hw,
            },
            f"checkpoints_fmb/pidnet_ep{ep:03d}.pth"
        )

        print(f"[SAVE] fusion_ep{ep:03d}.pth + pidnet_ep{ep:03d}.pth")


if __name__ == "__main__":
    train(
        ir_dir="",
        vi_dir="",
        mask_dir="",
        depth_ckpt_path="",
        num_classes=15,
        hw=(600, 800),
        batch_size=2,
        epochs=15000,
        lr_fusion=2e-4,
        lr_seg=1e-3,
        lambda_l1=1.0,
        lambda_grad=5.0,
        lambda_seg=1.0,
        seg_warmup_epochs=5,
        pidnet_augment=True,
        use_detail_loss=False,
        device="",

        dino_model_name="convnext_base.dinov3_lvd1689m",
        depth_pool_Ms=(0, 0, 0, 0),


        fusion_resume_path="",
        pidnet_resume_path="",
    )
