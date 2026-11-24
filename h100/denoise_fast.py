"""Immutable π0.5 action expert for fixed ten-step, per-bucket graph inference."""

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


def linear_weight(layer):
    weight = layer.weight.float()
    if hasattr(layer, "lora_B"):
        weight = weight + layer.lora_B.float() @ layer.lora_A.float()
    return weight.bfloat16().contiguous()


def dense(layer, x):
    return F.linear(x, layer[0], layer[1])


def projection(layer):
    bias = None if layer.bias is None else layer.bias.bfloat16()
    return linear_weight(layer), bias


class Denoiser:
    def __init__(self, model, obs, noise, cache, times):
        self.cache = cache
        self.input = projection(model.action_in_proj)
        self.output = projection(model.action_out_proj)
        self.condition = [model.embed_suffix(obs, noise, t)[3] for t in times]
        self.layers = []
        self.kv = []
        batch, length, _ = noise.shape
        self.offsets = torch.arange(length, device=noise.device)
        mask, kv = cache
        self.prefix_length = mask.shape[1]
        self.mask = F.pad(mask, (0, length), value=True)
        # Efficient attention requires the row stride of an additive bias to be aligned.
        padded = (self.mask.size(1) + 7) // 8 * 8
        self.bias = torch.zeros(
            (batch, 1, 1, padded), device=noise.device, dtype=torch.bfloat16
        )
        self.bias = self.bias[..., : self.mask.size(1)]
        for layer, (k, v) in zip(model.llm.layers, kv, strict=True):
            attn = layer.attn
            h, d = attn.num_heads, attn.head_dim
            self.heads, self.dim = h, d
            assert k.shape[2] == 1 and v.shape == k.shape
            kbuf = torch.empty(
                (batch, 1, k.size(1) + length, d), device=k.device, dtype=torch.bfloat16
            )
            self.kv.append((kbuf, torch.empty_like(kbuf)))
            qkv = torch.cat(
                [
                    linear_weight(proj[1])
                    for proj in (attn.q_proj, attn.k_proj, attn.v_proj)
                ]
            )
            mlp = layer.mlps[1]
            scale = mlp.lora_config.scaling_value
            gate_up = (
                mlp.w_gating.float()
                + scale * (mlp.w_gating_lora_a.float() @ mlp.w_gating_lora_b.float())
            ).to(torch.bfloat16)
            gate_up = gate_up.transpose(1, 2).reshape(-1, mlp.features).contiguous()
            down = (
                (
                    mlp.w_linear.float()
                    + scale
                    * (mlp.w_linear_lora_a.float() @ mlp.w_linear_lora_b.float())
                )
                .T.bfloat16()
                .contiguous()
            )
            self.layers.append(
                (
                    self.modulation(layer.pre_attention_norms[1]),
                    qkv,
                    linear_weight(attn.o_proj[1]),
                    self.modulation(layer.pre_ffw_norms[1]),
                    gate_up,
                    down,
                )
            )
        self.final = self.modulation(model.llm.final_norms[1])
        self.freq = 10000 ** (
            torch.arange(self.dim // 2, device=noise.device, dtype=torch.float32)
            * 2
            / self.dim
        )

    def modulation(self, norm):
        m = norm.ada_modulation
        values = []
        for cond in self.condition:
            scale, shift, gate = m(cond.to(m.weight.dtype)).unsqueeze(1).chunk(3, -1)
            values.append(
                (
                    (1 + scale.float()).bfloat16(),
                    shift.bfloat16().contiguous(),
                    gate.bfloat16().contiguous(),
                )
            )
        return values

    def prepare(self):
        mask, kv = self.cache
        p = self.prefix_length
        self.mask[:, :p].copy_(mask)
        self.bias.zero_().masked_fill_(~self.mask[:, None, None, :], float("-inf"))
        angle = (mask.sum(-1)[:, None] + self.offsets)[..., None, None] / self.freq
        for (k, v), (kb, vb) in zip(kv, self.kv, strict=True):
            kb[:, :, :p].copy_(k.transpose(1, 2))
            vb[:, :, :p].copy_(v.transpose(1, 2))
        return angle.cos().bfloat16(), angle.sin().bfloat16()

    @staticmethod
    def rope(x, cos, sin):
        a, b = x.chunk(2, -1)
        return torch.cat((a * cos - b * sin, b * cos + a * sin), -1)

    @staticmethod
    def norm(x, mod):
        return torch.addcmul(mod[1], F.rms_norm(x, (x.size(-1),), eps=1e-6), mod[0])

    def ten(self, noise, fused=True, steps=10, prepared=None):
        import denoise_kernels

        norm = denoise_kernels.norm if fused else self.norm
        cos, sin = self.prepare() if prepared is None else prepared
        x = noise.bfloat16()
        b, t, _ = x.shape
        p = self.prefix_length
        with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
            for j in range(steps):
                y = dense(self.input, x)
                for (am, qkv, out, fm, up, down), (kb, vb) in zip(
                    self.layers, self.kv, strict=True
                ):
                    z = F.linear(norm(y, am[j]), qkv).reshape(
                        b, t, self.heads + 2, self.dim
                    )
                    if fused:
                        q = denoise_kernels.qkv(z, cos, sin, kb, vb)
                    else:
                        zq, zk, zv = z.split([self.heads, 1, 1], 2)
                        kb[:, :, p:].copy_(self.rope(zk, cos, sin).transpose(1, 2))
                        vb[:, :, p:].copy_(zv.transpose(1, 2))
                        q = self.rope(zq, cos, sin).transpose(1, 2)
                    a = F.scaled_dot_product_attention(
                        q,
                        kb.expand(-1, self.heads, -1, -1),
                        vb.expand(-1, self.heads, -1, -1),
                        attn_mask=self.bias,
                    )
                    y = torch.addcmul(
                        y, F.linear(a.transpose(1, 2).reshape(b, t, -1), out), am[j][2]
                    )
                    u, v = F.linear(norm(y, fm[j]), up).chunk(2, -1)
                    y = torch.addcmul(y, F.linear(F.gelu(u) * v, down), fm[j][2])
                x = x - 0.1 * dense(self.output, norm(y, self.final[j]))
        return x.float()
