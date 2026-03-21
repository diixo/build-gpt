'''
Use RoPE:

* in encoder self-attention
* in decoder self-attention

CrossAttention without RoPE
'''

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from model_llama import RMSNorm, RotaryEmbedding, CausalSelfAttention

@dataclass
class Seq2SeqConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    flash_attn: bool = True
    model_type: str = ""

    rope_base: float = 10000.0
    use_rope: bool = True

    mlp_bias: bool = False


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_dim = 4 * config.n_embd

        self.c_fc1 = nn.Linear(config.n_embd, hidden_dim, bias=config.mlp_bias)
        self.c_fc2 = nn.Linear(config.n_embd, hidden_dim, bias=config.mlp_bias)
        self.silu = nn.SiLU()
        self.c_proj = nn.Linear(hidden_dim, config.n_embd, bias=config.mlp_bias)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x1 = self.c_fc1(x)
        x2 = self.c_fc2(x)
        hidden = self.silu(x1) * x2
        return self.c_proj(hidden)


class SelfAttention(nn.Module):
    """
    Bidirectional self-attention for encoder.
    """
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.flash_attn = config.flash_attn
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

        self.use_rope = config.use_rope
        if self.use_rope:
            self.rope = RotaryEmbedding(
                dim=self.head_dim,
                base=config.rope_base,
                max_seq_len=config.block_size
            )


    def forward(self, x, attention_mask=None):
        B, T, C = x.size()

        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)

        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, nh, T, hs)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        if self.use_rope:
            q = self.rope.apply_rotary(q, T)
            k = self.rope.apply_rotary(k, T)

        if attention_mask is not None:
            assert attention_mask.shape == (B, T)
            # True = masked
            attn_mask = (attention_mask == 0)[:, None, None, :].to(device=x.device, dtype=torch.bool)
        else:
            attn_mask = None

        if self.flash_attn:
            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                is_causal=False
            )
        else:
            attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B, nh, T, T)

            if attn_mask is not None:
                attn = attn.masked_fill(attn_mask, float("-inf"))

            attn = F.softmax(attn, dim=-1)
            y = attn @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y


class CrossAttention(nn.Module):
    """
    Decoder attends to encoder hidden states.
    q <- decoder
    k,v <- encoder
    """
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.flash_attn = config.flash_attn
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head

        self.q_proj = nn.Linear(config.n_embd, config.n_embd)
        self.k_proj = nn.Linear(config.n_embd, config.n_embd)
        self.v_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1


    def forward(self, x, encoder_hidden_states, encoder_attention_mask=None):
        """
        x: (B, T_dec, C)
        encoder_hidden_states: (B, T_enc, C)
        encoder_attention_mask: (B, T_enc), 1=real token, 0=pad
        """
        B, T_dec, C = x.size()
        B2, T_enc, C2 = encoder_hidden_states.size()
        assert B == B2 and C == C2

        q = self.q_proj(x)
        k = self.k_proj(encoder_hidden_states)
        v = self.v_proj(encoder_hidden_states)

        q = q.view(B, T_dec, self.n_head, self.head_dim).transpose(1, 2)   # (B, nh, T_dec, hs)
        k = k.view(B, T_enc, self.n_head, self.head_dim).transpose(1, 2)   # (B, nh, T_enc, hs)
        v = v.view(B, T_enc, self.n_head, self.head_dim).transpose(1, 2)

        if encoder_attention_mask is not None:
            assert encoder_attention_mask.shape == (B, T_enc)
            kv_mask = (encoder_attention_mask == 0)[:, None, None, :].to(device=x.device, dtype=torch.bool)
        else:
            kv_mask = None

        if self.flash_attn:
            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=kv_mask,
                is_causal=False
            )
        else:
            attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B, nh, T_dec, T_enc)

            if kv_mask is not None:
                attn = attn.masked_fill(kv_mask, float("-inf"))

            attn = F.softmax(attn, dim=-1)
            y = attn @ v

        y = y.transpose(1, 2).contiguous().view(B, T_dec, C)
        y = self.c_proj(y)
        return y


class EncoderBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = RMSNorm(config.n_embd)
        self.attn = SelfAttention(config)
        self.ln_2 = RMSNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x, attention_mask=None):
        x = x + self.attn(self.ln_1(x), attention_mask=attention_mask)
        x = x + self.mlp(self.ln_2(x))
        return x


class DecoderBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = RMSNorm(config.n_embd)
        self.self_attn = CausalSelfAttention(config)

        self.ln_cross = RMSNorm(config.n_embd)
        self.cross_attn = CrossAttention(config)

        self.ln_2 = RMSNorm(config.n_embd)
        self.mlp = MLP(config)


    def forward(
        self,
        x,
        encoder_hidden_states,
        decoder_attention_mask=None,
        encoder_attention_mask=None,
    ):
        x = x + self.self_attn(self.ln_1(x), attention_mask=decoder_attention_mask)
        x = x + self.cross_attn(
            self.ln_cross(x),
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
        )
        x = x + self.mlp(self.ln_2(x))
        return x


@dataclass
class Seq2SeqOutput:
    logits: torch.Tensor
    loss: Optional[torch.Tensor] = None
    encoder_hidden_states: Optional[torch.Tensor] = None


class Seq2SeqTransformer(nn.Module):
    def __init__(self, config: Seq2SeqConfig = None, **kwargs):
        super().__init__()
        if config is None:
            config = Seq2SeqConfig(**kwargs)
        self.config = config

        # Shared token embedding
        self.shared_wte = nn.Embedding(config.vocab_size, config.n_embd)

        # Positional embeddings only when RoPE is disabled
        self.encoder_wpe = None if config.use_rope else nn.Embedding(config.block_size, config.n_embd)
        self.decoder_wpe = None if config.use_rope else nn.Embedding(config.block_size, config.n_embd)

        self.encoder = nn.ModuleDict(dict(
            h=nn.ModuleList([EncoderBlock(config) for _ in range(config.n_layer)]),
            ln_f=RMSNorm(config.n_embd),
        ))

        self.decoder = nn.ModuleDict(dict(
            h=nn.ModuleList([DecoderBlock(config) for _ in range(config.n_layer)]),
            ln_f=RMSNorm(config.n_embd),
        ))

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # tie weights
        self.shared_wte.weight = self.lm_head.weight

        self.apply(self._init_weights)


    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, "NANOGPT_SCALE_INIT"):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


    def encode(self, encoder_input_ids, encoder_attention_mask=None):
        B, T = encoder_input_ids.size()
        assert T <= self.config.block_size, f"Encoder seq len {T} > block_size {self.config.block_size}"

        x = self.shared_wte(encoder_input_ids)

        if not self.config.use_rope:
            pos = torch.arange(0, T, dtype=torch.long, device=encoder_input_ids.device)
            x = x + self.encoder_wpe(pos)

        for block in self.encoder["h"]:
            x = block(x, attention_mask=encoder_attention_mask)

        x = self.encoder["ln_f"](x)
        return x


    def decode(
        self,
        decoder_input_ids,
        encoder_hidden_states,
        decoder_attention_mask=None,
        encoder_attention_mask=None,
    ):
        B, T = decoder_input_ids.size()
        assert T <= self.config.block_size, f"Decoder seq len {T} > block_size {self.config.block_size}"

        x = self.shared_wte(decoder_input_ids)

        if not self.config.use_rope:
            pos = torch.arange(0, T, dtype=torch.long, device=decoder_input_ids.device)
            x = x + self.decoder_wpe(pos)

        for block in self.decoder["h"]:
            x = block(
                x,
                encoder_hidden_states=encoder_hidden_states,
                decoder_attention_mask=decoder_attention_mask,
                encoder_attention_mask=encoder_attention_mask,
            )

        x = self.decoder["ln_f"](x)
        logits = self.lm_head(x)
        return logits


    def forward(
        self,
        encoder_input_ids,
        decoder_input_ids,
        encoder_attention_mask=None,
        decoder_attention_mask=None,
        targets=None,
    ):
        encoder_hidden_states = self.encode(
            encoder_input_ids=encoder_input_ids,
            encoder_attention_mask=encoder_attention_mask,
        )

        logits = self.decode(
            decoder_input_ids=decoder_input_ids,
            encoder_hidden_states=encoder_hidden_states,
            decoder_attention_mask=decoder_attention_mask,
            encoder_attention_mask=encoder_attention_mask,
        )

        loss = None
        if targets is not None:
            targets = targets.clone()

            if decoder_attention_mask is not None:
                targets[decoder_attention_mask == 0] = -100

            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-100
            )

        return Seq2SeqOutput(
            logits=logits,
            loss=loss,
            encoder_hidden_states=encoder_hidden_states
        )
