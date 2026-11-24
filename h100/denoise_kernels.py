"""Row kernels for the fixed-shape Gemma action expert."""

import torch
import triton
import triton.language as tl


@triton.jit
def _norm(X, A, B, Y, T: tl.constexpr, D: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, D)
    x = tl.load(X + row * D + col).to(tl.float32)
    param = (row // T) * D + col
    normalized = x * tl.rsqrt(tl.sum(x * x, axis=0) / D + 1.0e-6)
    tl.store(Y + row * D + col, normalized * tl.load(A + param) + tl.load(B + param))


def norm(x, mod):
    batch, t, d = x.shape
    y = torch.empty_like(x)
    _norm[(batch * t,)](x, mod[0].contiguous(), mod[1].contiguous(), y, t, d)
    return y


@triton.jit
def _qkv(
    Z, C, S, Q, K, V, H: tl.constexpr, T: tl.constexpr, D: tl.constexpr, P: tl.constexpr
):
    row = tl.program_id(0)
    h = tl.program_id(1)
    b = row // T
    t = row % T
    i = tl.arange(0, D // 2)
    pos = (row * (H + 2) + h) * D + i
    a = tl.load(Z + pos)
    tail = tl.load(Z + pos + D // 2)
    dest = ((b * H + h) * T + t) * D + i
    slot = (b * (P + T) + P + t) * D + i
    if h < H + 1:
        c = tl.load(C + row * (D // 2) + i)
        s = tl.load(S + row * (D // 2) + i)
        u = a * c - tail * s
        v = tail * c + a * s
        if h < H:
            tl.store(Q + dest, u)
            tl.store(Q + dest + D // 2, v)
        else:
            tl.store(K + slot, u)
            tl.store(K + slot + D // 2, v)
    else:
        tl.store(V + slot, a)
        tl.store(V + slot + D // 2, tail)


def qkv(z, cos, sin, k, v):
    b, t, h2, d = z.shape
    h = h2 - 2
    q = torch.empty((b, h, t, d), dtype=z.dtype, device=z.device)
    _qkv[(b * t, h2)](z, cos, sin, q, k, v, h, t, d, k.size(2) - t)
    return q
