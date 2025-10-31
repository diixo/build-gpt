"""
GPT model:

1) Rotary Position Embeddings (RoPE) - roformer implemented.
2) rotary_pct ( float , optional, defaults to 0.25) — percentage of hidden dimensions to allocate to rotary embeddings
3) QK-normalization when using RoPE.
4) RMSNorm instead of LayerNorm.
5) Weight tying option (tie_word_embeddings) between token embeddings and LM head.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional


@dataclass
class GPTConfig:
    block_size: int = 1024  # max sequence length
    vocab_size: int = 50257 # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    n_layer: int = 12       # number of layers
    n_head: int = 12        # number of heads
    n_embd: int = 768       # embedding dimension
    flash_attn: bool = True # whether to use flash attention (scaled_dot_product_attention)

    # RoPE params
    rope_base: float = 10000.0  # standard base (θ). For learning on length=2048 may use 10000.0
    use_rope: bool = True       # whether to use RoPE or not
    rotary_pct: float = 0.25    # percentage of head_dim to apply RoPE to (1.0 = all)
    tie_word_embeddings: bool = True    # whether to tie word embeddings and LM head weights


def norm(x, eps=1e-5):
    # Purely functional rmsnorm with no learnable params
    return F.rms_norm(x, (x.size(-1),), eps=eps)

@dataclass
class GPTOutput:
    logits: torch.Tensor
    loss: Optional[torch.Tensor] = None


class RotaryEmbedding(nn.Module):
    """
    Implements precomputed Rotary Position Embedding (RoPE) cache for efficiency.
    """

    def __init__(self, dim: int, base: float = 10000.0, max_seq_len: int = 2048, rotary_pct: float = 1.0):
        """
        Args:
            dim: Head dimension (hs)
            base: RoPE base theta (usually 10000.0)
            max_seq_len: Maximum sequence length
        """
        super().__init__()
        assert dim % 2 == 0, "Head dimension must be even for RoPE"
        self.dim = dim
        self.base = base
        self.max_seq_len = max_seq_len
        self.rotary_dim = int(dim * rotary_pct)

        # precompute frequencies
        half_dim = self.rotary_dim // 2

        channel_range = 2 * torch.arange(0, half_dim, dtype=torch.float32)
        inv_freq = 1.0 / (base ** (channel_range / dim))

        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)                # (T, half_dim)
        freqs_cos = torch.cos(freqs)[None, None, :, :]  # (1, 1, T, half_dim)
        freqs_sin = torch.sin(freqs)[None, None, :, :]  # (1, 1, T, half_dim)

        # store as buffers (moved automatically with model.to(device))
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)


    def apply_rotary(self, x: torch.Tensor, seq_len: int):
        """
        Applies rotary transformation to the first rotary_dim of tensor x.
        Expects x.shape = (B, n_head, T, head_dim)
        """
        assert x.ndim == 4, f"Expected 4D tensor (B, nH, T, d), got: {x.shape}"
        cos = self.freqs_cos[:, :, :seq_len, :].to(x.device)
        sin = self.freqs_sin[:, :, :seq_len, :].to(x.device)
        rotary_part = x[..., :self.rotary_dim]
        non_rotary_part = x[..., self.rotary_dim:]

        #d = x.shape[3] // 2
        d = self.rotary_dim // 2
        x1, x2 = rotary_part[..., :d], rotary_part[..., d:]

        y1 = x1 * cos + x2 * sin
        y2 = x1 * (-sin) + x2 * cos
        rotated = torch.cat([y1, y2], dim=-1)
        return torch.cat([rotated, non_rotary_part], dim=-1)


class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.flash_attn = config.flash_attn

        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd

        # RoPE config
        self.use_rope = config.use_rope
        self.rope_base = config.rope_base
        self.rotary_pct = config.rotary_pct

        if self.use_rope:
            head_dim = config.n_embd // config.n_head
            self.rope = RotaryEmbedding(
                dim=head_dim,
                base=self.rope_base,
                max_seq_len=config.block_size,
                rotary_pct=self.rotary_pct
            )

            # positional embedding для non-RoPE части
            if self.rotary_pct < 1.0:
                self.wpe = nn.Embedding(config.block_size, head_dim - self.rope.rotary_dim)
            else:
                self.wpe = None
        else:
            self.wpe = nn.Embedding(config.block_size, config.n_embd)


    def forward(self, x, pos=None):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        # e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*hs=C=768 channels in the Transformer
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        if self.use_rope:
            q = self.rope.apply_rotary(q, T)
            k = self.rope.apply_rotary(k, T)

        # --- add WPE only to non-rotation part ---
        if self.wpe is not None and self.use_rope and self.rotary_pct < 1.0:
            rotary_dim = self.rope.rotary_dim
            nonrotary_dim = (C // self.n_head) - rotary_dim
            if nonrotary_dim > 0:
                # self.wpe(pos) -> (T, nonrotary_dim)
                # [None, None, :, :] -> (1, 1, T, nonrotary_dim)
                pos_emb = self.wpe(pos)[None, None, :, :]
                q[..., rotary_dim:] += pos_emb
                k[..., rotary_dim:] += pos_emb
        elif self.wpe is not None and not self.use_rope:
            # classic GPT-2 positional embeddings
            pos_emb = self.wpe(pos)
            x = x + pos_emb

        if self.use_rope:
            q = norm(q)
            k = norm(k)

        if self.n_head == 1 or not self.flash_attn:
            # manual attention implementation
            att = (q @ k.transpose(-2, -1)) / math.sqrt(k.size(-1)) # (B, nh, T, T)
            att = att.masked_fill(torch.triu(torch.ones(T, T, device=x.device), 1).bool(), float('-inf'))
            att = F.softmax(att, dim=-1)
            y = att @ v  # (B, nh, T, hs)
        else:
            # use PyTorch flash attention (scaled_dot_product_attention)
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True) # flash attention

        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu    = nn.GELU(approximate='tanh')
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    # head_layer+1 ​= head_layer + Attn(LN(head_layer)) + MLP(LN(head_layer))

    def __init__(self, config):
        super().__init__()
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)

    def forward(self, x, pos=None):
        x = x + self.attn(norm(x), pos=pos)
        x = x + self.mlp(norm(x))
        return x


class GPTNeoX(nn.Module):
    def __init__(self, config=None, **kwargs):
        super().__init__()
        if config is None:
            config = GPTConfig(**kwargs)
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = None if config.use_rope and config.rotary_pct == 1.0 else nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd)
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        if config.tie_word_embeddings:
            # weight sharing scheme, tie_word_embeddings = True
            self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)


    def forward(self, idx, targets=None):
        # idx is of shape (B, T)
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        # forward the token and posisition embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape (T)

        tok_emb = self.transformer.wte(idx) # token embeddings of shape (B, T, n_embd)
        if self.transformer.wpe is not None:
            pos_emb = self.transformer.wpe(pos)
            x = tok_emb + pos_emb
        else:
            x = tok_emb

        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x, pos=pos)
        # forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return GPTOutput(logits=logits, loss=loss)


    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding and not self.use_rope:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params
