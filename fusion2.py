# fusion_pidnet_full.py
from __future__ import annotations

from typing import List, Tuple, Optional, Sequence, Dict, Union
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# DepthAnythingV2 encoder extraction file (your "depthv2.py")
from depthv2 import (
    DINOv2,
    load_depthanything_encoder_weights,
    extract_12_layers,
)

# ============================================================
# Utils
# ============================================================

def maybe_to_float01(x: torch.Tensor) -> torch.Tensor:
    if x.dtype.is_floating_point:
        if x.max() > 1.5:
            return x / 255.0
        return x
    return x.float() / 255.0

def to_3ch(x: torch.Tensor) -> torch.Tensor:
    """(B,1,H,W)->(B,3,H,W) or keep (B,3,H,W)."""
    if x.dim() != 4:
        raise ValueError(f"Expect BCHW, got {tuple(x.shape)}")
    if x.size(1) == 1:
        return x.repeat(1, 3, 1, 1)
    if x.size(1) == 3:
        return x
    raise ValueError(f"Expect 1 or 3 channels, got C={x.size(1)}")

def resize_bchw(x: torch.Tensor, hw: Tuple[int, int]) -> torch.Tensor:
    """BCHW bilinear resize; if already target size, no-op."""
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
    """
    Pad only right/bottom so output becomes target_hw.
    x: (B,C,H,W)
    """
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
    """Crop back to original (undo right/bottom padding)."""
    H, W = hw
    return x[..., :H, :W]

def _largest_divisor_leq(n: int, max_div: int) -> int:
    for g in range(min(max_div, n), 0, -1):
        if n % g == 0:
            return g
    return 1

# ============================================================
# 1) Frozen Feature Extractors
# ============================================================

import timm
from timm.data import resolve_model_data_config

class FrozenDinoV3ConvNeXtFeatures(nn.Module):
    """
    Frozen timm DINOv3 ConvNeXt features (4 stages).

    Return:
      feats_list: list len=4, each (B, C_i, H_i, W_i)
      strides: list len=4, e.g. [4,8,16,32]
      channels: list len=4, e.g. [128,256,512,1024] for convnext_base
    """
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

        # feature_info provides channels and stride reductions
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

        feats = self.model(x)  # list of BCHW
        return list(feats), list(self.strides), list(self.channels)

class FrozenDepthAnythingEncoderTokens(nn.Module):
    """
    Frozen DepthAnythingV2 encoder (DINOv2) intermediate patch tokens.

    Return:
      feats: list len=12, each (B, N_patch, C)
      patch_grid: (H_patch, W_patch)
    """
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

        feats = extract_12_layers(self.encoder, x, with_cls=False)  # tuple len=12
        feats = list(feats)
        gh, gw = self.img_hw[0] // self.patch, self.img_hw[1] // self.patch
        return feats, (gh, gw)

# ============================================================
# 2) Depth-guided "Hypergraph" via Node->Depth Cross-Attention (no M=16)
# ============================================================

# ============================================================
# 2) Hypergraph blocks (original style):
#    depth tokens -> M hyperedges -> cosine similarity assign -> node->edge->node
# ============================================================

class DepthToHyperedges(nn.Module):
    """Depth tokens -> M hyperedge embeddings via cross-attention (learnable queries)."""
    def __init__(self, c_d: int, M: int = 16, d_model: int = 256, heads: int = 8, dropout: float = 0.0):
        super().__init__()
        assert d_model % heads == 0
        self.M = int(M)
        self.kv_proj = nn.Linear(c_d, d_model) if c_d != d_model else nn.Identity()
        self.ln_kv = nn.LayerNorm(d_model)
        self.q = nn.Parameter(torch.randn(1, self.M, d_model) * 0.02)
        self.attn = nn.MultiheadAttention(d_model, heads, dropout=dropout, batch_first=True)

    def forward(self, X_d: torch.Tensor) -> torch.Tensor:
        """
        X_d: (B, Nd, Cd)
        return E: (B, M, d_model)
        """
        B, _, _ = X_d.shape
        KV = self.ln_kv(self.kv_proj(X_d))        # (B,Nd,D)
        Q  = self.q.expand(B, -1, -1)             # (B,M,D)
        E, _ = self.attn(Q, KV, KV, need_weights=False)
        return E                                  # (B,M,D)


class NodeToHyperedgeAssign(nn.Module):
    """
    A(B,N,M) = softmax( cosine(node,edge) / tau )
    """
    def __init__(self, Cn: int, Ce: int, d: int = 256, tau_init: float = 0.07, learnable_tau: bool = True):
        super().__init__()
        self.q_proj = nn.Linear(Cn, d)
        self.k_proj = nn.Linear(Ce, d)
        if learnable_tau:
            self.log_tau = nn.Parameter(torch.log(torch.tensor(float(tau_init))))
        else:
            self.register_buffer("log_tau", torch.log(torch.tensor(float(tau_init))))

    def forward(self, X_n: torch.Tensor, E: torch.Tensor):
        """
        X_n: (B,N,Cn)
        E:   (B,M,Ce)
        """
        Q = F.normalize(self.q_proj(X_n), dim=-1)      # (B,N,d)
        K = F.normalize(self.k_proj(E), dim=-1)        # (B,M,d)
        logits = Q @ K.transpose(-1, -2)               # (B,N,M)
        tau = torch.exp(self.log_tau).clamp(min=1e-4)
        A = torch.softmax(logits / tau, dim=-1)
        return A, logits


class HypergraphFuseLayer(nn.Module):
    """
    node->edge (degree-normalized) -> fuse with depth edges -> edge->node
    + residual scales (avoid oversmoothing)
    + optional top-k sparsify for locality
    """
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
        """
        X_n: (B,N,Cn)
        E:   (B,M,Ce)
        return:
          X_out: (B,N,Cn)
          E_out: (B,M,d)   (edge feature updated in hidden dim d)
          A:     (B,N,M)
        """
        A, _ = self.assign(X_n, E)            # (B,N,M)
        A = self._sparsify(A)

        Xd = self.node_proj(X_n)              # (B,N,d)
        At = A.transpose(1, 2)                # (B,M,N)

        # node->edge degree-normalized mean
        denom = At.sum(dim=-1, keepdim=True) + 1e-6
        H_from_nodes = (At @ Xd) / denom      # (B,M,d)

        Ed = self.edge_proj(E)                # (B,M,d)
        delta_e = self.edge_fuse(torch.cat([Ed, H_from_nodes], dim=-1))
        E_out = Ed + self.edge_scale * delta_e

        # edge->node
        X_up = A @ E_out                      # (B,N,d)
        X_up = self.back_proj(X_up)           # (B,N,Cn)

        X_out = self.norm(X_n + self.node_scale * X_up)
        return X_out, E_out, A


class DepthGuidedHypergraphBlock(nn.Module):
    """
    原始超图方式封装：
      depth tokens -> M hyperedges
      node tokens  -> assign(A) -> hypergraph fuse (可堆叠多层)
    """
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
            Ce = hg_hidden  # edge dim becomes d after first layer
        self.hg_layers = nn.ModuleList(layers)

    def forward(self, x_n: torch.Tensor, x_d: torch.Tensor) -> torch.Tensor:
        """
        x_n: (B,N,Cn)
        x_d: (B,Nd,Cd)
        """
        E = self.depth2edge(x_d)              # (B,M,De)
        for hg in self.hg_layers:
            x_n, E, _ = hg(x_n, E)
        return x_n


# ============================================================
# 3) Cross-modal fusion (DepthForge-style) - token版保持不变，但每个尺度各一套
# ============================================================

class CrossModalForgeFusion(nn.Module):
    """
    gate base + token dictionary attention + small-scale residual injection
    """
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
            gate = torch.sigmoid(self.gate_mlp(h))  # (B,N,1)

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
# 4) Multi-Scale Decoder (改为 2D 特征版，但逻辑沿用你原来的 Token-DRD 解码器)
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
    """
    输入：4层 fused feature maps (B,Ci,Hi,Wi)  [对应 stride4/8/16/32]
    输出：fused gray logits at out_hw
    逻辑沿用你原 Token-DRD:
      - 每尺度对齐到 embed_dim
      - 选择 fuse_hw
      - resize+gate+concat -> gamma融合
      - 逐级 upsample + gated skip
      - 再额外从 stride4 up 到 out_hw（避免一步插值造成网格/糊）
    """
    def __init__(
        self,
        in_channels: Sequence[int],
        embed_dim: int = 256,
        fuse_to: str = "middle",      # "lowest"/"middle"/"highest"
        out_channels: int = 1,
        use_transformer: bool = False,  # 默认关掉，避免位置编码周期纹理
    ):
        super().__init__()
        self.in_channels = list(in_channels)
        self.num_scales = len(self.in_channels)
        assert self.num_scales == 4, "Expect 4 scales from ConvNeXt"
        assert fuse_to in ["lowest", "middle", "highest"]
        self.fuse_to = fuse_to
        self.embed_dim = embed_dim
        self.use_transformer = use_transformer

        # align each scale to embed_dim
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

        # upsample blocks between scales (最多 3 次)
        self.up_blocks = nn.ModuleList([UpFuseBlock(embed_dim) for _ in range(self.num_scales - 1)])

        # 从最大尺度(stride4) -> out_hw 的额外两级上采样（stride2, stride1）
        self.post_up1 = UpFuseBlock(embed_dim)
        self.post_up2 = UpFuseBlock(embed_dim)

        self.final_refine = nn.Sequential(
            ConvGNAct(embed_dim, embed_dim, 3, 1, 1),
            ConvGNAct(embed_dim, embed_dim // 2, 3, 1, 1),
        )
        self.out_head = nn.Conv2d(embed_dim // 2, out_channels, 1)

    def pick_fuse_hw(self, hw_list: List[Tuple[int, int]]) -> Tuple[int, int]:
        areas = [h * w for h, w in hw_list]
        order = sorted(range(len(hw_list)), key=lambda i: areas[i])  # low->high
        if self.fuse_to == "lowest":
            return hw_list[order[0]]
        if self.fuse_to == "highest":
            return hw_list[order[-1]]
        return hw_list[order[len(order) // 2]]  # middle

    def forward(self, feats_list: List[torch.Tensor], out_hw: Tuple[int, int]) -> torch.Tensor:
        assert len(feats_list) == self.num_scales

        # align
        maps = []
        hw_list = []
        for i, f in enumerate(feats_list):
            fa = self.align[i](f)  # (B,D,Hi,Wi)
            maps.append(fa)
            hw_list.append(fa.shape[-2:])

        # build skip dict
        skip_dict: Dict[Tuple[int, int], torch.Tensor] = {}
        for f, hw in zip(maps, hw_list):
            skip_dict[hw] = f if hw not in skip_dict else 0.5 * (skip_dict[hw] + f)

        fuse_hw = self.pick_fuse_hw(hw_list)

        # resize to fuse_hw + gate
        resized = []
        for i, f in enumerate(maps):
            fr = f if f.shape[-2:] == fuse_hw else F.interpolate(f, size=fuse_hw, mode="bilinear", align_corners=False)
            fr = self.scale_gates[i](fr)
            resized.append(fr)

        x = self.gamma(torch.cat(resized, dim=1))  # (B,D,Hf,Wf)

        # upsample to larger scales (from fuse_hw -> ... -> max hw among inputs)
        unique_hw = sorted(set(hw_list + [fuse_hw]), key=lambda hw: hw[0] * hw[1])  # low->high
        start = unique_hw.index(fuse_hw)

        step = 0
        for nxt in unique_hw[start + 1:]:
            if step >= len(self.up_blocks):
                break
            x = self.up_blocks[step](x, skip_dict.get(nxt, None), nxt)
            step += 1

        # 现在 x 通常在最大尺度（stride4 对齐后的大小）
        # 再两级上采样到 out_hw，避免一步插值
        if x.shape[-2:] != out_hw:
            mid_hw = (out_hw[0] // 2, out_hw[1] // 2)
            if x.shape[-2:] != mid_hw:
                x = self.post_up1(x, None, mid_hw)
            if x.shape[-2:] != out_hw:
                x = self.post_up2(x, None, out_hw)

        feat = self.final_refine(x)
        logits = self.out_head(feat)  # (B,1,H,W)
        return logits

# ============================================================
# 5) Fusion Backbone + Full FusionNet (with internal padding)
# ============================================================

class IRVIS_HypergraphFusionBackbone(nn.Module):
    """
    IR/VI -> frozen DINOv3 ConvNeXt feature maps (nodes)
          -> frozen DepthAnything tokens (hyperedges)
          -> per-scale depth-guided hypergraph (node cross-attend depth tokens)
          -> per-scale cross-modal fusion
    return: fused_feats_list (len=4), where each is (B,Ci,Hi,Wi)
    """
    def __init__(
        self,
        depth_ckpt_path: str,
        train_hw: Tuple[int, int] = (448, 448),

        # DINOv3 CNN
        dino_model_name: str = "convnext_base.dinov3_lvd1689m",
        dino_out_indices: Sequence[int] = (0, 1, 2, 3),

        # DepthAnything 12层 -> 4层选择（浅->深）
        depth_layer_ids_4: Sequence[int] = (2, 5, 8, 11),

        # depth-guided hypergraph (cross-attn)
        hg_d_model: int = 256,
        hg_heads: int = 8,
        hg_layers: int = 1,
        hg_dropout: float = 0.0,

        # 可选：对 depth tokens 轻度压缩（控制显存/速度），默认 0=不压缩
        # 例如你可设 (256,256,0,0)
        depth_pool_Ms: Sequence[int] = (0, 0, 0, 0),

        # cross-modal fusion
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

        # Frozen feature extractors
        self.dino = FrozenDinoV3ConvNeXtFeatures(
            model_name=dino_model_name,
            out_indices=dino_out_indices,
            device=device,
        )
        self.depth = FrozenDepthAnythingEncoderTokens(
            ckpt_path=depth_ckpt_path,
            img_hw=train_hw,  # will be overwritten to work_hw below
            device=device,
        )

        # pad multiple: lcm(depth_patch=14, convnext_stem_stride=4) = 28  (小得多，不会像224那样夸张)
        self.pad_multiple = _lcm(int(self.depth.patch), 4)
        self.work_hw = ceil_hw_to_multiple(self.train_hw, self.pad_multiple)

        # Make DepthAnything extractor resize to work_hw
        self.depth.img_hw = self.work_hw

        depth_dim = int(self.depth.embed_dim)
        node_dims = list(self.dino.channels)  # 4 scales, e.g. [128,256,512,1024]

        # depth token pools (optional)
        # 这里把 depth_pool_Ms 复用成 “每个尺度的超边数 M”
        # 为了兼容你现有脚本默认 [0,0,0,0]，当 M<=0 时默认用 16（不会把超图关掉）
        self.hyper_Ms = [int(M) if (M is not None and int(M) > 0) else 16 for M in self.depth_pool_Ms]

        # 保持 forward 结构不变：depth_pools 仍然存在，但这里统一 Identity（不再做 token pool）
        self.depth_pools = nn.ModuleList([nn.Identity() for _ in range(4)])

        # per-scale hypergraph blocks（原始超图：depth->M edges->assign->node update）
        self.hg_blocks = nn.ModuleList([
            DepthGuidedHypergraphBlock(
                node_dim=node_dims[i],
                depth_dim=depth_dim,
                hyper_M=self.hyper_Ms[i],
                hyper_d_model=hg_d_model,
                hyper_heads=hg_heads,
                hg_hidden=hg_d_model,  # 为了不引入新参数，这里 hidden = hg_d_model
                hg_layers=hg_layers,
                hg_sparse_topk=4,  # 固定 4（等同原始代码默认）
                dropout=hg_dropout,
            )
            for i in range(4)
        ])

        # per-scale fusion
        self.fusion = nn.ModuleList([
            CrossModalForgeFusion(node_dims[i], token_length=fusion_dict_tokens, gate_type="token")
            for i in range(4)
        ])

        self.node_dims = node_dims

    def train(self, mode: bool = True):
        super().train(mode)
        # keep frozen backbones eval
        self.dino.eval()
        self.depth.eval()
        return self

    @staticmethod
    def feat_to_tokens(feat: torch.Tensor) -> torch.Tensor:
        # (B,C,H,W) -> (B,HW,C)
        return feat.flatten(2).transpose(1, 2).contiguous()

    @staticmethod
    def tokens_to_feat(tokens: torch.Tensor, hw: Tuple[int, int]) -> torch.Tensor:
        # (B,HW,C) -> (B,C,H,W)
        B, N, C = tokens.shape
        H, W = hw
        if N != H * W:
            raise ValueError(f"N={N} must equal H*W={H*W}")
        return tokens.transpose(1, 2).reshape(B, C, H, W).contiguous()

    def forward(self, ir: torch.Tensor, vi: torch.Tensor):
        """
        ir, vi: (B,1,H,W) or (B,3,H,W)
        Internally:
          - resize to train_hw
          - pad to work_hw (multiple of 28)
          - extract ConvNeXt features at work_hw
          - extract DepthAnything tokens at work_hw
        """
        ir = maybe_to_float01(ir)
        vi = maybe_to_float01(vi)

        ir = resize_bchw(ir, self.train_hw)
        vi = resize_bchw(vi, self.train_hw)

        ir_w = pad_bchw_to_hw(ir, self.work_hw, mode=self.pad_mode)
        vi_w = pad_bchw_to_hw(vi, self.work_hw, mode=self.pad_mode)

        # frozen extractors
        with torch.no_grad():
            dino_ir_feats, _, _ = self.dino(ir_w, hw=self.work_hw)  # 4 scales
            dino_vi_feats, _, _ = self.dino(vi_w, hw=self.work_hw)

            dep_ir_all, _ = self.depth(ir_w)  # 12 layers tokens
            dep_vi_all, _ = self.depth(vi_w)

        fused_feats_list: List[torch.Tensor] = []

        for i in range(4):
            f_ir = dino_ir_feats[i]  # (B,Ci,Hi,Wi)
            f_vi = dino_vi_feats[i]
            Hi, Wi = f_ir.shape[-2:]

            # node tokens
            Xn_ir = self.feat_to_tokens(f_ir)  # (B,Ni,Ci)
            Xn_vi = self.feat_to_tokens(f_vi)

            # choose 4 depth layers: shallow->deep mapping
            lid = self.depth_layer_ids_4[i]
            Xd_ir = dep_ir_all[lid]  # (B,Nd,Cd)
            Xd_vi = dep_vi_all[lid]

            # optional pool (Identity if M<=0)
            Xd_ir_m = self.depth_pools[i](Xd_ir)
            Xd_vi_m = self.depth_pools[i](Xd_vi)

            # depth-guided hypergraph update (node cross-attend depth tokens)
            Xn_ir = self.hg_blocks[i](Xn_ir, Xd_ir_m)
            Xn_vi = self.hg_blocks[i](Xn_vi, Xd_vi_m)

            # cross-modal fusion
            Xf = self.fusion[i](Xn_ir, Xn_vi)  # (B,Ni,Ci)

            # back to feature map
            fused_map = self.tokens_to_feat(Xf, (Hi, Wi))  # (B,Ci,Hi,Wi)
            fused_feats_list.append(fused_map)

        return fused_feats_list  # 4 scales at work_hw

class IRVISFusionNet(nn.Module):
    """
    Full fusion model:
      backbone -> 4-scale fused feature maps -> MultiScaleFeatureDRDDecoder -> fused gray
    Output cropped back to train_hw.
    """
    def __init__(
        self,
        depth_ckpt_path: str,
        train_hw: Tuple[int, int] = (448, 448),

        dino_model_name: str = "convnext_base.dinov3_lvd1689m",
        depth_layer_ids_4: Sequence[int] = (2, 5, 8, 11),

        # depth-guided hypergraph
        hg_d_model: int = 256,
        hg_heads: int = 8,
        hg_layers: int = 1,
        hg_dropout: float = 0.0,
        depth_pool_Ms: Sequence[int] = (0, 0, 0, 0),  # 可改(256,256,0,0)

        # fusion + decoder
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

        # decoder input channels = ConvNeXt stage channels
        in_ch = self.backbone.node_dims
        self.decoder = MultiScaleFeatureDRDDecoder(
            in_channels=in_ch,            # [C4,C8,C16,C32] order is out_indices order
            embed_dim=decoder_embed_dim,
            fuse_to=decoder_fuse_to,
            out_channels=1,
            use_transformer=False,
        )

    def train(self, mode: bool = True):
        super().train(mode)
        # backbone 内部会保持 frozen extractor eval
        return self

    def forward(self, ir: torch.Tensor, vi: torch.Tensor) -> torch.Tensor:
        fused_feats = self.backbone(ir, vi)  # 4-scale maps at work_hw
        logits_w = self.decoder(fused_feats, out_hw=self.backbone.work_hw)
        fused_w = torch.sigmoid(logits_w)  # (B,1,Hwork,Wwork)
        fused = crop_bchw_to_hw(fused_w, self.train_hw)  # (B,1,Htrain,Wtrain)
        return fused

# ============================================================
# 6) Wrap with PIDNet for joint training
# ============================================================

class FusionPIDNetSystem(nn.Module):
    """
    Joint system: fused image -> PIDNet -> segmentation logits/list.
    PIDNet is TRAINABLE here.
    """
    def __init__(self, fusion_net: IRVISFusionNet, pidnet: nn.Module):
        super().__init__()
        self.fusion = fusion_net
        self.pidnet = pidnet

    def forward(self, ir: torch.Tensor, vi: torch.Tensor):
        fused = self.fusion(ir, vi)       # (B,1,Htrain,Wtrain)
        fused3 = fused.repeat(1, 3, 1, 1) # PIDNet expects 3ch
        seg_out = self.pidnet(fused3)
        return fused, seg_out
