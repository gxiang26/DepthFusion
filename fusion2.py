
from __future__ import annotations

from typing import List, Tuple, Optional, Sequence, Dict, Union
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


from depthv2 import (
    DINOv2,
    load_depthanything_encoder_weights,
    extract_12_layers,
)

# ============================================================
# ============================================================

def maybe_to_float01(x: torch.Tensor) -> torch.Tensor:
    if x.dtype.is_floating_point:
        if x.max() > 1.5:
            return x / 255.0
        return x
    return x.float() / 255.0

def to_3ch(x: torch.Tensor) -> torch.Tensor:

    if x.dim() != 4:
        raise ValueError(f"Expect BCHW, got {tuple(x.shape)}")
    if x.size(1) == 1:
        return x.repeat(1, 3, 1, 1)
    if x.size(1) == 3:
        return x
    raise ValueError(f"Expect 1 or 3 channels, got C={x.size(1)}")

def resize_bchw(x: torch.Tensor, hw: Tuple[int, int]) -> torch.Tensor:

    if x.shape[-2:] == hw:
        return x
    return F.interpolate(x, size=hw, mode="bilinear", align_corners=False)

def normalize_chw(x: torch.Tensor, mean, std) -> torch.Tensor:
    device, dtype = x.device, x.dtype
    mean_t = torch.tensor(mean, device=device, dtype=dtype).view(1, 3, 1, 1)
    std_t  = torch.tensor(std,  device=device, dtype=dtype).view(1, 3, 1, 1)
    return (x - mean_t) / std_t

def _lcm(a: int, b: int) -> int:
    return a * b // math.gcd(a, b)

def ceil_hw_to_multiple(hw: Tuple[int, int], multiple: int) -> Tuple[int, int]:
    H, W = hw
    H2 = ((H + multiple - 1) // multiple) * multiple
    W2 = ((W + multiple - 1) // multiple) * multiple
    return (H2, W2)

def pad_bchw_to_hw(x: torch.Tensor, target_hw: Tuple[int, int], mode: str = "replicate") -> torch.Tensor:

    Ht, Wt = target_hw
    H, W = x.shape[-2:]
    pad_h = Ht - H
    pad_w = Wt - W
    if pad_h < 0 or pad_w < 0:
        raise ValueError(f"target_hw must be >= input hw, got target={target_hw}, input={(H, W)}")
    if pad_h == 0 and pad_w == 0:
        return x
    return F.pad(x, (0, pad_w, 0, pad_h), mode=mode)

def crop_bchw_to_hw(x: torch.Tensor, hw: Tuple[int, int]) -> torch.Tensor:

    H, W = hw
    return x[..., :H, :W]

def _largest_divisor_leq(n: int, max_div: int) -> int:
    for g in range(min(max_div, n), 0, -1):
        if n % g == 0:
            return g
    return 1

# ============================================================
# ============================================================

import timm
from timm.data import resolve_model_data_config

class FrozenDinoV3ConvNeXtFeatures(nn.Module):

    def __init__(
        self,
        model_name: str = "convnext_base.dinov3_lvd1689m",
        pretrained: bool = True,
        weights_path: Optional[str] = None,
        out_indices: Sequence[int] = (0, 1, 2, 3),
        device: Optional[Union[str, torch.device]] = None,
    ):
        super().__init__()
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=tuple(out_indices),
        )

        if weights_path is not None and weights_path != "":
            sd = torch.load(weights_path, map_location="cpu")
            if isinstance(sd, dict) and "state_dict" in sd:
                sd = sd["state_dict"]
            if isinstance(sd, dict):
                sd = {k.replace("module.", ""): v for k, v in sd.items()}
            self.model.load_state_dict(sd, strict=False)

        self.model.eval()
        if device is not None:
            self.model.to(device)

        for p in self.model.parameters():
            p.requires_grad_(False)

        # data config
        try:
            self.data_cfg = resolve_model_data_config(self.model)
            self.mean = self.data_cfg.get("mean", (0.485, 0.456, 0.406))
            self.std  = self.data_cfg.get("std",  (0.229, 0.224, 0.225))
        except Exception:
            self.mean = (0.485, 0.456, 0.406)
            self.std  = (0.229, 0.224, 0.225)


        fi = getattr(self.model, "feature_info", None)
        if fi is None:
            raise RuntimeError("features_only model must have feature_info.")
        self.channels = list(fi.channels())
        self.strides  = list(fi.reduction())

    def train(self, mode: bool = True):
        super().train(False)
        self.model.eval()
        return self

    @torch.no_grad()
    def forward(self, x: torch.Tensor, hw: Tuple[int, int]):
        x = maybe_to_float01(x)
        x = to_3ch(x)
        x = resize_bchw(x, hw)
        x = normalize_chw(x, self.mean, self.std)

        feats = self.model(x)
        return list(feats), list(self.strides), list(self.channels)

class FrozenDepthAnythingEncoderTokens(nn.Module):

    def __init__(
        self,
        ckpt_path: str,
        dino_name: str = "vitb",
        img_hw: Tuple[int, int] = (448, 448),
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        device: Optional[Union[str, torch.device]] = None,
    ):
        super().__init__()
        if device is None:
            device = "cpu"
        if not ckpt_path:
            raise ValueError("DepthAnything encoder needs ckpt_path.")

        self.encoder = DINOv2(dino_name)
        self.encoder = load_depthanything_encoder_weights(self.encoder, ckpt_path, device=str(device))
        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad_(False)

        self.img_hw = img_hw
        self.mean = mean
        self.std = std
        self.patch = getattr(self.encoder, "patch_size", 14)
        self.embed_dim = getattr(self.encoder, "embed_dim", getattr(self.encoder, "num_features", None))
        if self.embed_dim is None:
            raise RuntimeError("Cannot infer embed_dim for DepthAnything encoder.")

    def train(self, mode: bool = True):
        super().train(False)
        self.encoder.eval()
        return self

    @torch.no_grad()
    def forward(self, x: torch.Tensor):
        x = maybe_to_float01(x)
        x = to_3ch(x)
        x = resize_bchw(x, self.img_hw)
        x = normalize_chw(x, self.mean, self.std)

        feats = extract_12_layers(self.encoder, x, with_cls=False)
        feats = list(feats)
        gh, gw = self.img_hw[0] // self.patch, self.img_hw[1] // self.patch
        return feats, (gh, gw)

# ============================================================
# ============================================================

class DepthToHyperedges(nn.Module):

    def __init__(self, c_d: int, M: int = 16, d_model: int = 256, heads: int = 8, dropout: float = 0.0):
        super().__init__()
        assert d_model % heads == 0
        self.M = int(M)
        self.kv_proj = nn.Linear(c_d, d_model) if c_d != d_model else nn.Identity()
        self.ln_kv = nn.LayerNorm(d_model)
        self.q = nn.Parameter(torch.randn(1, self.M, d_model) * 0.02)
        self.attn = nn.MultiheadAttention(d_model, heads, dropout=dropout, batch_first=True)

    def forward(self, X_d: torch.Tensor) -> torch.Tensor:

        B, _, _ = X_d.shape
        KV = self.ln_kv(self.kv_proj(X_d))
        Q  = self.q.expand(B, -1, -1)
        E, _ = self.attn(Q, KV, KV, need_weights=False)
        return E


class NodeToHyperedgeAssign(nn.Module):

    def __init__(self, Cn: int, Ce: int, d: int = 256, tau_init: float = 0.07, learnable_tau: bool = True):
        super().__init__()
        self.q_proj = nn.Linear(Cn, d)
        self.k_proj = nn.Linear(Ce, d)
        if learnable_tau:
            self.log_tau = nn.Parameter(torch.log(torch.tensor(float(tau_init))))
        else:
            self.register_buffer("log_tau", torch.log(torch.tensor(float(tau_init))))

    def forward(self, X_n: torch.Tensor, E: torch.Tensor):

        Q = F.normalize(self.q_proj(X_n), dim=-1)
        K = F.normalize(self.k_proj(E), dim=-1)
        logits = Q @ K.transpose(-1, -2)
        tau = torch.exp(self.log_tau).clamp(min=1e-4)
        A = torch.softmax(logits / tau, dim=-1)
        return A, logits


class HypergraphFuseLayer(nn.Module):

    def __init__(self, Cn: int, Ce: int, d: int = 256, sparse_topk: int = 0, dropout: float = 0.0):
        super().__init__()
        self.assign = NodeToHyperedgeAssign(Cn, Ce, d=d, tau_init=0.07, learnable_tau=True)

        self.node_proj = nn.Linear(Cn, d) if Cn != d else nn.Identity()
        self.edge_proj = nn.Linear(Ce, d) if Ce != d else nn.Identity()

        self.edge_fuse = nn.Sequential(
            nn.LayerNorm(2 * d),
            nn.Linear(2 * d, 2 * d),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * d, d),
        )

        self.edge_scale = nn.Parameter(torch.tensor(1e-3))
        self.node_scale = nn.Parameter(torch.tensor(1e-3))

        self.back_proj = nn.Linear(d, Cn) if d != Cn else nn.Identity()
        self.norm = nn.LayerNorm(Cn)

        self.sparse_topk = int(sparse_topk) if sparse_topk is not None else 0

    def _sparsify(self, A: torch.Tensor) -> torch.Tensor:
        if self.sparse_topk <= 0:
            return A
        k = min(self.sparse_topk, A.shape[-1])
        _, idx = torch.topk(A, k=k, dim=-1)
        mask = torch.zeros_like(A)
        mask.scatter_(-1, idx, 1.0)
        A = A * mask
        A = A / (A.sum(dim=-1, keepdim=True) + 1e-6)
        return A

    def forward(self, X_n: torch.Tensor, E: torch.Tensor):

        A, _ = self.assign(X_n, E)
        A = self._sparsify(A)

        Xd = self.node_proj(X_n)
        At = A.transpose(1, 2)

        # node->edge degree-normalized mean
        denom = At.sum(dim=-1, keepdim=True) + 1e-6
        H_from_nodes = (At @ Xd) / denom

        Ed = self.edge_proj(E)
        delta_e = self.edge_fuse(torch.cat([Ed, H_from_nodes], dim=-1))
        E_out = Ed + self.edge_scale * delta_e

        # edge->node
        X_up = A @ E_out
        X_up = self.back_proj(X_up)

        X_out = self.norm(X_n + self.node_scale * X_up)
        return X_out, E_out, A


class DepthGuidedHypergraphBlock(nn.Module):

    def __init__(
        self,
        node_dim: int,
        depth_dim: int,
        hyper_M: int = 16,
        hyper_d_model: int = 256,
        hyper_heads: int = 8,
        hg_hidden: int = 256,
        hg_layers: int = 1,
        hg_sparse_topk: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hyper_M = int(hyper_M)
        self.depth2edge = DepthToHyperedges(depth_dim, M=self.hyper_M, d_model=hyper_d_model, heads=hyper_heads, dropout=dropout)

        layers = []
        Ce = hyper_d_model
        for _ in range(int(hg_layers)):
            layers.append(HypergraphFuseLayer(node_dim, Ce, d=hg_hidden, sparse_topk=hg_sparse_topk, dropout=dropout))
            Ce = hg_hidden
        self.hg_layers = nn.ModuleList(layers)

    def forward(self, x_n: torch.Tensor, x_d: torch.Tensor) -> torch.Tensor:

        E = self.depth2edge(x_d)
        for hg in self.hg_layers:
            x_n, E, _ = hg(x_n, E)
        return x_n


# ============================================================
# ============================================================

class CrossModalForgeFusion(nn.Module):

    def __init__(
        self,
        embed_dims: int,
        token_length: int = 64,
        gate_type: str = "token",
        use_softmax: bool = True,
        scale_init: float = 1e-3,
        alpha_learnable: bool = True,
        dropout: float = 0.0,
        normalize_dict: bool = True,
    ):
        super().__init__()
        self.C = embed_dims
        self.M = token_length
        self.gate_type = gate_type
        self.use_softmax = use_softmax
        self.normalize_dict = normalize_dict

        self.learnable_tokens = nn.Parameter(torch.empty(self.M, self.C))
        nn.init.uniform_(self.learnable_tokens, -1.0 / math.sqrt(self.C), 1.0 / math.sqrt(self.C))

        self.norm_ir = nn.LayerNorm(self.C)
        self.norm_vi = nn.LayerNorm(self.C)
        self.norm_f0 = nn.LayerNorm(self.C)

        self.mlp_token2feat = nn.Linear(self.C, self.C)
        self.mlp_delta = nn.Sequential(
            nn.LayerNorm(self.C),
            nn.Linear(self.C, self.C),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.C, self.C),
        )

        self.scale = nn.Parameter(torch.tensor(scale_init))
        self.drop_attn = nn.Dropout(dropout)

        if gate_type == "global":
            self.gate_mlp = nn.Sequential(nn.Linear(2 * self.C, self.C), nn.GELU(), nn.Linear(self.C, 1))
        elif gate_type == "token":
            self.gate_mlp = nn.Sequential(nn.Linear(2 * self.C, self.C), nn.GELU(), nn.Linear(self.C, 1))
        else:
            raise ValueError("gate_type must be 'global' or 'token'")

        if alpha_learnable:
            self.alpha_logits = nn.Parameter(torch.zeros(3))
        else:
            self.register_buffer("alpha_logits", torch.zeros(3))

    def forward(self, x_ir: torch.Tensor, x_vi: torch.Tensor) -> torch.Tensor:
        B, N, C = x_ir.shape
        assert x_vi.shape == (B, N, C)

        x_irn = self.norm_ir(x_ir)
        x_vin = self.norm_vi(x_vi)

        if self.gate_type == "global":
            h = torch.cat([x_vin.mean(dim=1), x_irn.mean(dim=1)], dim=-1)
            gate = torch.sigmoid(self.gate_mlp(h)).view(B, 1, 1)
        else:
            h = torch.cat([x_vin, x_irn], dim=-1)
            gate = torch.sigmoid(self.gate_mlp(h))

        x_f0 = gate * x_vin + (1.0 - gate) * x_irn
        x_f0 = self.norm_f0(x_f0)

        T = self.learnable_tokens
        Tn = F.normalize(T, dim=-1) if self.normalize_dict else T

        scale = C ** -0.5
        attn_f  = torch.einsum("bnc,mc->bnm", x_f0,  Tn) * scale
        attn_ir = torch.einsum("bnc,mc->bnm", x_irn, Tn) * scale
        attn_vi = torch.einsum("bnc,mc->bnm", x_vin, Tn) * scale

        alpha = torch.softmax(self.alpha_logits, dim=0)
        attn = alpha[0] * attn_f + alpha[1] * attn_ir + alpha[2] * attn_vi
        if self.use_softmax:
            attn = F.softmax(attn, dim=-1)
        attn = self.drop_attn(attn)

        delta = torch.einsum("bnm,mc->bnc", attn, self.mlp_token2feat(T))
        delta = self.mlp_delta(delta + x_f0)

        return x_f0 + self.scale * delta

# ============================================================
# ============================================================

class AttentionGate2D(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        mid = max(channels // reduction, 1)
        self.ca = nn.Sequential(
            nn.Conv2d(channels, mid, 1), nn.GELU(),
            nn.Conv2d(mid, channels, 1), nn.Sigmoid()
        )
        self.sa = nn.Sequential(nn.Conv2d(channels, 1, 3, padding=1), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x

class ConvGNAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, s: int = 1, p: int = 1, gn_max_groups: int = 32):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, bias=False)
        g = _largest_divisor_leq(out_ch, gn_max_groups)
        self.gn = nn.GroupNorm(g, out_ch)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.gn(self.conv(x)))

class GatedSkipFuse(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1),
            nn.GELU(),
            nn.Conv2d(channels, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        g = self.gate(torch.cat([x, s], dim=1))
        return x + g * s

class UpFuseBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.pre = ConvGNAct(channels, channels, 3, 1, 1)
        self.fuse = GatedSkipFuse(channels)
        self.post = nn.Sequential(
            ConvGNAct(channels, channels, 3, 1, 1),
            ConvGNAct(channels, channels, 3, 1, 1),
        )

    def forward(self, x: torch.Tensor, skip: Optional[torch.Tensor], out_hw: Tuple[int, int]) -> torch.Tensor:
        x = F.interpolate(x, size=out_hw, mode="bilinear", align_corners=False)
        x = self.pre(x)
        if skip is not None:
            x = self.fuse(x, skip)
        x = self.post(x)
        return x

class MultiScaleFeatureDRDDecoder(nn.Module):

    def __init__(
        self,
        in_channels: Sequence[int],
        embed_dim: int = 256,
        fuse_to: str = "middle",
        out_channels: int = 1,
        use_transformer: bool = False,
    ):
        super().__init__()
        self.in_channels = list(in_channels)
        self.num_scales = len(self.in_channels)
        assert self.num_scales == 4, "Expect 4 scales from ConvNeXt"
        assert fuse_to in ["lowest", "middle", "highest"]
        self.fuse_to = fuse_to
        self.embed_dim = embed_dim
        self.use_transformer = use_transformer


        self.align = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c, embed_dim, kernel_size=1, bias=False),
                nn.GroupNorm(_largest_divisor_leq(embed_dim, 32), embed_dim),
                nn.GELU(),
            )
            for c in self.in_channels
        ])

        self.scale_gates = nn.ModuleList([AttentionGate2D(embed_dim) for _ in range(self.num_scales)])

        self.gamma = nn.Sequential(
            ConvGNAct(embed_dim * self.num_scales, embed_dim, 3, 1, 1),
            ConvGNAct(embed_dim, embed_dim, 3, 1, 1),
        )


        self.up_blocks = nn.ModuleList([UpFuseBlock(embed_dim) for _ in range(self.num_scales - 1)])


        self.post_up1 = UpFuseBlock(embed_dim)
        self.post_up2 = UpFuseBlock(embed_dim)

        self.final_refine = nn.Sequential(
            ConvGNAct(embed_dim, embed_dim, 3, 1, 1),
            ConvGNAct(embed_dim, embed_dim // 2, 3, 1, 1),
        )
        self.out_head = nn.Conv2d(embed_dim // 2, out_channels, 1)

    def pick_fuse_hw(self, hw_list: List[Tuple[int, int]]) -> Tuple[int, int]:
        areas = [h * w for h, w in hw_list]
        order = sorted(range(len(hw_list)), key=lambda i: areas[i])
        if self.fuse_to == "lowest":
            return hw_list[order[0]]
        if self.fuse_to == "highest":
            return hw_list[order[-1]]
        return hw_list[order[len(order) // 2]]

    def forward(self, feats_list: List[torch.Tensor], out_hw: Tuple[int, int]) -> torch.Tensor:
        assert len(feats_list) == self.num_scales


        maps = []
        hw_list = []
        for i, f in enumerate(feats_list):
            fa = self.align[i](f)
            maps.append(fa)
            hw_list.append(fa.shape[-2:])


        skip_dict: Dict[Tuple[int, int], torch.Tensor] = {}
        for f, hw in zip(maps, hw_list):
            skip_dict[hw] = f if hw not in skip_dict else 0.5 * (skip_dict[hw] + f)

        fuse_hw = self.pick_fuse_hw(hw_list)


        resized = []
        for i, f in enumerate(maps):
            fr = f if f.shape[-2:] == fuse_hw else F.interpolate(f, size=fuse_hw, mode="bilinear", align_corners=False)
            fr = self.scale_gates[i](fr)
            resized.append(fr)

        x = self.gamma(torch.cat(resized, dim=1))


        unique_hw = sorted(set(hw_list + [fuse_hw]), key=lambda hw: hw[0] * hw[1])
        start = unique_hw.index(fuse_hw)

        step = 0
        for nxt in unique_hw[start + 1:]:
            if step >= len(self.up_blocks):
                break
            x = self.up_blocks[step](x, skip_dict.get(nxt, None), nxt)
            step += 1


        if x.shape[-2:] != out_hw:
            mid_hw = (out_hw[0] // 2, out_hw[1] // 2)
            if x.shape[-2:] != mid_hw:
                x = self.post_up1(x, None, mid_hw)
            if x.shape[-2:] != out_hw:
                x = self.post_up2(x, None, out_hw)

        feat = self.final_refine(x)
        logits = self.out_head(feat)
        return logits

# ============================================================
# ============================================================

class IRVIS_HypergraphFusionBackbone(nn.Module):

    def __init__(
        self,
        depth_ckpt_path: str,
        train_hw: Tuple[int, int] = (448, 448),


        dino_model_name: str = "convnext_base.dinov3_lvd1689m",
        dino_out_indices: Sequence[int] = (0, 1, 2, 3),


        depth_layer_ids_4: Sequence[int] = (2, 5, 8, 11),


        hg_d_model: int = 256,
        hg_heads: int = 8,
        hg_layers: int = 1,
        hg_dropout: float = 0.0,



        depth_pool_Ms: Sequence[int] = (0, 0, 0, 0),


        fusion_dict_tokens: int = 64,

        pad_mode: str = "replicate",
        device: Optional[Union[str, torch.device]] = None,
    ):
        super().__init__()
        self.train_hw = train_hw
        self.pad_mode = pad_mode

        self.depth_layer_ids_4 = list(depth_layer_ids_4)
        assert len(self.depth_layer_ids_4) == 4, "Need 4 depth layer ids."
        assert all(0 <= i < 12 for i in self.depth_layer_ids_4), "Depth layer ids must be in [0,11]."

        self.depth_pool_Ms = list(depth_pool_Ms)
        assert len(self.depth_pool_Ms) == 4


        self.dino = FrozenDinoV3ConvNeXtFeatures(
            model_name=dino_model_name,
            out_indices=dino_out_indices,
            device=device,
        )
        self.depth = FrozenDepthAnythingEncoderTokens(
            ckpt_path=depth_ckpt_path,
            img_hw=train_hw,
            device=device,
        )


        self.pad_multiple = _lcm(int(self.depth.patch), 4)
        self.work_hw = ceil_hw_to_multiple(self.train_hw, self.pad_multiple)


        self.depth.img_hw = self.work_hw

        depth_dim = int(self.depth.embed_dim)
        node_dims = list(self.dino.channels)


        self.hyper_Ms = [int(M) if (M is not None and int(M) > 0) else 16 for M in self.depth_pool_Ms]


        self.depth_pools = nn.ModuleList([nn.Identity() for _ in range(4)])


        self.hg_blocks = nn.ModuleList([
            DepthGuidedHypergraphBlock(
                node_dim=node_dims[i],
                depth_dim=depth_dim,
                hyper_M=self.hyper_Ms[i],
                hyper_d_model=hg_d_model,
                hyper_heads=hg_heads,
                hg_hidden=hg_d_model,
                hg_layers=hg_layers,
                hg_sparse_topk=4,
                dropout=hg_dropout,
            )
            for i in range(4)
        ])


        self.fusion = nn.ModuleList([
            CrossModalForgeFusion(node_dims[i], token_length=fusion_dict_tokens, gate_type="token")
            for i in range(4)
        ])

        self.node_dims = node_dims

    def train(self, mode: bool = True):
        super().train(mode)

        self.dino.eval()
        self.depth.eval()
        return self

    @staticmethod
    def feat_to_tokens(feat: torch.Tensor) -> torch.Tensor:

        return feat.flatten(2).transpose(1, 2).contiguous()

    @staticmethod
    def tokens_to_feat(tokens: torch.Tensor, hw: Tuple[int, int]) -> torch.Tensor:

        B, N, C = tokens.shape
        H, W = hw
        if N != H * W:
            raise ValueError(f"N={N} must equal H*W={H*W}")
        return tokens.transpose(1, 2).reshape(B, C, H, W).contiguous()

    def forward(self, ir: torch.Tensor, vi: torch.Tensor):

        ir = maybe_to_float01(ir)
        vi = maybe_to_float01(vi)

        ir = resize_bchw(ir, self.train_hw)
        vi = resize_bchw(vi, self.train_hw)

        ir_w = pad_bchw_to_hw(ir, self.work_hw, mode=self.pad_mode)
        vi_w = pad_bchw_to_hw(vi, self.work_hw, mode=self.pad_mode)


        with torch.no_grad():
            dino_ir_feats, _, _ = self.dino(ir_w, hw=self.work_hw)
            dino_vi_feats, _, _ = self.dino(vi_w, hw=self.work_hw)

            dep_ir_all, _ = self.depth(ir_w)
            dep_vi_all, _ = self.depth(vi_w)

        fused_feats_list: List[torch.Tensor] = []

        for i in range(4):
            f_ir = dino_ir_feats[i]
            f_vi = dino_vi_feats[i]
            Hi, Wi = f_ir.shape[-2:]


            Xn_ir = self.feat_to_tokens(f_ir)
            Xn_vi = self.feat_to_tokens(f_vi)


            lid = self.depth_layer_ids_4[i]
            Xd_ir = dep_ir_all[lid]
            Xd_vi = dep_vi_all[lid]


            Xd_ir_m = self.depth_pools[i](Xd_ir)
            Xd_vi_m = self.depth_pools[i](Xd_vi)


            Xn_ir = self.hg_blocks[i](Xn_ir, Xd_ir_m)
            Xn_vi = self.hg_blocks[i](Xn_vi, Xd_vi_m)


            Xf = self.fusion[i](Xn_ir, Xn_vi)


            fused_map = self.tokens_to_feat(Xf, (Hi, Wi))
            fused_feats_list.append(fused_map)

        return fused_feats_list

class IRVISFusionNet(nn.Module):

    def __init__(
        self,
        depth_ckpt_path: str,
        train_hw: Tuple[int, int] = (448, 448),

        dino_model_name: str = "convnext_base.dinov3_lvd1689m",
        depth_layer_ids_4: Sequence[int] = (2, 5, 8, 11),


        hg_d_model: int = 256,
        hg_heads: int = 8,
        hg_layers: int = 1,
        hg_dropout: float = 0.0,
        depth_pool_Ms: Sequence[int] = (0, 0, 0, 0),

        fusion_dict_tokens: int = 64,
        decoder_embed_dim: int = 256,
        decoder_fuse_to: str = "middle",

        device: Optional[Union[str, torch.device]] = None,
    ):
        super().__init__()
        self.train_hw = train_hw

        self.backbone = IRVIS_HypergraphFusionBackbone(
            depth_ckpt_path=depth_ckpt_path,
            train_hw=train_hw,
            dino_model_name=dino_model_name,
            depth_layer_ids_4=depth_layer_ids_4,
            hg_d_model=hg_d_model,
            hg_heads=hg_heads,
            hg_layers=hg_layers,
            hg_dropout=hg_dropout,
            depth_pool_Ms=depth_pool_Ms,
            fusion_dict_tokens=fusion_dict_tokens,
            device=device,
        )


        in_ch = self.backbone.node_dims
        self.decoder = MultiScaleFeatureDRDDecoder(
            in_channels=in_ch,
            embed_dim=decoder_embed_dim,
            fuse_to=decoder_fuse_to,
            out_channels=1,
            use_transformer=False,
        )

    def train(self, mode: bool = True):
        super().train(mode)

        return self

    def forward(self, ir: torch.Tensor, vi: torch.Tensor) -> torch.Tensor:
        fused_feats = self.backbone(ir, vi)
        logits_w = self.decoder(fused_feats, out_hw=self.backbone.work_hw)
        fused_w = torch.sigmoid(logits_w)
        fused = crop_bchw_to_hw(fused_w, self.train_hw)
        return fused

# ============================================================
# ============================================================

class FusionPIDNetSystem(nn.Module):

    def __init__(self, fusion_net: IRVISFusionNet, pidnet: nn.Module):
        super().__init__()
        self.fusion = fusion_net
        self.pidnet = pidnet

    def forward(self, ir: torch.Tensor, vi: torch.Tensor):
        fused = self.fusion(ir, vi)
        fused3 = fused.repeat(1, 3, 1, 1)
        seg_out = self.pidnet(fused3)
        return fused, seg_out
