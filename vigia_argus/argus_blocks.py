# ultralytics/nn/modules/argus_blocks.py
# Argus-V8X custom blocks: SimAM + tiny Swin block (export-friendly)

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------
#  SimAM (CVPR'21) – parameter-free attention
# ------------------------------
class SimAM(nn.Module):
    """
    Simple Attention Module (SimAM).
    - Parameter-free (no learnable weights)
    - ONNX/TFLite friendly (elementwise ops only)
    Paper: "SimAM: A Simple, Parameter-Free Attention Module for CNNs"
    """

    def __init__(self, e_lambda: float = 1e-4):
        super().__init__()
        self.e_lambda = float(e_lambda)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        mu = x.mean(dim=(2, 3), keepdim=True)           # channel-wise mean
        x_mu = x - mu
        den = 4.0 * (x_mu.pow(2).mean(dim=(2, 3), keepdim=True)) + self.e_lambda
        attn = torch.sigmoid(x_mu.pow(2) / den + 0.5)   # same shape as x
        return x * attn


# ------------------------------
#  Tiny Swin block (no shift) for deep stage P5
# ------------------------------
# --- replace your SwinBlock + _SwinLayer with these ---

class SwinBlock(nn.Module):
    """
    Tiny Swin Transformer stack for deep stage (P5), channel-agnostic.
    Lazily builds with the input channel count on first forward,
    so it works across YOLO width scales (n/s/m/l/x).
    Args (positional per YAML):
        channels: (ignored for embedding; kept for backward-compat)
        depth: number of transformer layers (1–2 recommended)
        window_size: e.g., 7
        num_heads: 3–4
        mlp_ratio: 3–4
        embed_dim: optional fixed dim (if set, uses 1x1 proj in/out)
    """

    def __init__(
        self,
        channels: int = None,
        depth: int = 1,
        window_size: int = 7,
        num_heads: int = 4,
        mlp_ratio: int = 4,
        embed_dim: int | None = None,
    ):
        super().__init__()
        # store config; don't build yet
        self.ws = int(window_size)
        self.depth = int(depth)
        self.num_heads = int(num_heads)
        self.mlp_ratio = int(mlp_ratio)

        # if user wants a fixed embed_dim, we’ll 1x1 project
        self.declared_dim = int(embed_dim) if embed_dim is not None else None

        # will be created on first forward based on input C
        self.embed_dim = None
        self.layers = None
        self.proj_in = nn.Identity()
        self.proj_out = nn.Identity()

    def _build(self, in_channels: int):
        # decide working dim
        dim = self.declared_dim if self.declared_dim is not None else in_channels

        # set projections if we force a different dim
        if dim != in_channels:
            self.proj_in = nn.Conv2d(in_channels, dim, kernel_size=1, bias=True)
            self.proj_out = nn.Conv2d(dim, in_channels, kernel_size=1, bias=True)

        # create transformer layers with the decided dim
        self.layers = nn.ModuleList([
            _SwinLayer(dim, self.ws, self.num_heads, self.mlp_ratio)
            for _ in range(self.depth)
        ])
        self.embed_dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # on first run (or after shape change), build with current C
        C = x.shape[1]
        if self.layers is None or (self.declared_dim is None and self.embed_dim != C):
            self._build(C)

        x = self.proj_in(x)
        for blk in self.layers:
            x = blk(x)
        x = self.proj_out(x)
        return x


class _SwinLayer(nn.Module):
    def __init__(self, dim: int, window_size: int, num_heads: int, mlp_ratio: int):
        super().__init__()
        self.ws = int(window_size)
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, self.ws, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        b, c, h, w = x.shape

        pad_h = (self.ws - h % self.ws) % self.ws
        pad_w = (self.ws - w % self.ws) % self.ws
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
        _, _, H, W = x.shape

        x_bhwc = x.permute(0, 2, 3, 1).contiguous()     # [B,H,W,C]
        win = window_partition(x_bhwc, self.ws)         # [Bn, ws, ws, C]
        Bn, ws, _, C = win.shape
        win = win.view(Bn, ws * ws, C)                  # [Bn, N, C]

        y = self.norm1(win)
        y = self.attn(y) + win
        z = self.mlp(self.norm2(y)) + y

        z = z.view(Bn, ws, ws, C)
        x_merged = window_reverse(z, self.ws, H, W)     # [B,H,W,C]
        out = x_merged.permute(0, 3, 1, 2).contiguous() # [B,C,H,W]

        if pad_h or pad_w:
            out = out[:, :, :h, :w]
        return out


class WindowAttention(nn.Module):
    """Multi-head self-attention inside a fixed window (no shift)."""
    def __init__(self, dim: int, window_size: int, num_heads: int):
        super().__init__()
        self.num_heads = int(num_heads)
        self.dim = int(dim)
        self.ws = int(window_size)
        head_dim = self.dim // self.num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(self.dim, self.dim * 3, bias=True)
        self.proj = nn.Linear(self.dim, self.dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [Bn, N, C], where N = ws*ws
        Bn, N, C = x.shape
        qkv = self.qkv(x).view(Bn, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)    # [3, Bn, heads, N, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale      # [Bn, heads, N, N]
        attn = attn.softmax(dim=-1)
        out = attn @ v                                     # [Bn, heads, N, head_dim]
        out = out.transpose(1, 2).contiguous().view(Bn, N, C)
        return self.proj(out)


class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


# ------------------------------
#  Window helpers (exportable)
# ------------------------------
def window_partition(x_bhwc: torch.Tensor, ws: int) -> torch.Tensor:
    """
    x_bhwc: [B, H, W, C] → windows [Bn, ws, ws, C]
    """
    B, H, W, C = x_bhwc.shape
    x = x_bhwc.view(B, H // ws, ws, W // ws, ws, C)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    return x.view(-1, ws, ws, C)


def window_reverse(windows: torch.Tensor, ws: int, H: int, W: int) -> torch.Tensor:
    """
    windows: [Bn, ws, ws, C] → [B, H, W, C]
    """
    B = int(windows.shape[0] // (H // ws * W // ws))
    x = windows.view(B, H // ws, W // ws, ws, ws, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    return x.view(B, H, W, -1)