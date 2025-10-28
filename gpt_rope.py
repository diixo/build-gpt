import inspect
import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass
from transformers import GPT2LMHeadModel


@dataclass
class GPTConfig:
    block_size: int = 1024  # max sequence length
    vocab_size: int = 50257 # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    n_layer: int = 12       # number of layers
    n_head: int = 12        # number of heads
    n_embd: int = 768       # embedding dimension
    bidirectional: bool = False # by GPT_LSTM using

    # RoPE params
    rope_base: float = 10000.0  # standard base (θ). For learning on length=2048 may use 10000.0
    rope_scale: float = 1.0     # scaling (if extend the model, for example: 2.0 for 1024->2048 NTK-scaling)


class RotaryEmbedding(nn.Module):
    """
    RoPE (rotary positional embeddings).
    Return cos, sin matrices forsequences length T, head_dim sizes.
    """

    def __init__(self, dim: int, base: float = 10000.0, scale: float = 1.0):
        """
        dim: head_dim (hs)
        base: θ (usually 10000.0)
        scale: scaling of positioning indices (for NTK-scaling)
        """
        super().__init__()
        assert dim % 2 == 0, "head_dim must be even for interleaved rotary implementation"
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        # inv_freq shape: (dim/2,)
        self.register_buffer("inv_freq", inv_freq)
        self.scale = float(scale)
        self.dim = dim


    def get_cos_sin(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        """
        Returns cos, sin tensors of forms: (seq_len, dim),
        ready for broadcast to: (B, nh, T, hs).
        """
        t = torch.arange(seq_len, device=device, dtype=dtype).float() / self.scale  # scaled positions
        # freqs: (T, dim/2)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq.to(dtype=dtype))
        # interleave to (T, dim)
        emb = torch.cat((freqs, freqs), dim=-1)  # (T, dim)
        cos = emb.cos()
        sin = emb.sin()
        return cos, sin

    @staticmethod
    def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        """
        x: (B, nh, T, hs)
        cos, sin: broadcastable to (1, 1, T, hs) or (T, hs)
        returns rotated x
        """
        # split even/odd
        # x[..., ::2] shape = (..., hs/2)
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        # rotate: (-x2, x1)
        x_rot = torch.stack((-x2, x1), dim=-1).reshape_as(x)
        # ensure cos/sin have trailing dims for broadcasting
        # cos, sin shape: (T, hs) -> make (1,1,T,hs)
        if cos.dim() == 2:
            cos = cos.unsqueeze(0).unsqueeze(0)
            sin = sin.unsqueeze(0).unsqueeze(0)
        return x * cos + x_rot * sin

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd

        # Rotary embedding instance (applies to q/k)
        self.rope = RotaryEmbedding(self.head_dim, base=config.rope_base, scale=config.rope_scale)


    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        # e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*hs=C=768 channels in the Transformer
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # --- apply RoPE to q and k ---
        cos, sin = self.rope.get_cos_sin(T, device=x.device, dtype=x.dtype)  # (T, hs)
        q = RotaryEmbedding.apply_rotary(q, cos, sin)
        k = RotaryEmbedding.apply_rotary(k, cos, sin)
        # ---------------------------------

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
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            #wpe = nn.Embedding(config.block_size, config.n_embd),
            # wpe removed, position embeddings coded by RoPE inside attention
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight

        # init params
        self.apply(self._init_weights)


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


    def forward(self, idx, targets=None):
        # idx is of shape (B, T)
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        # forward the token and posisition embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape (T)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (T, n_embd)
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (B, T, n_embd)
        x = tok_emb # + pos_emb
        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)
        # forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

