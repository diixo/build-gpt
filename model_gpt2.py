import math
import inspect
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from transformers import GPT2LMHeadModel
from typing import Optional


@dataclass
class GPTConfig:
    block_size: int = 1024  # max sequence length
    vocab_size: int = 50257 # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    n_layer: int = 12       # number of layers
    n_head: int = 12        # number of heads
    n_embd: int = 768       # embedding dimension
    flash_attn: bool = True # whether to use flash attention (scaled_dot_product_attention)
    model_type: str = ""    # model type


@dataclass
class GPTOutput:
    logits: torch.Tensor
    loss: Optional[torch.Tensor] = None


class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.flash_attn = config.flash_attn

        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd


    def forward(self, x, attention_mask=None):

        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        # e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*hs=C=768 channels in the Transformer
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # attention_mask: (B, T) -> (B, 1, 1, T) for broadcast

        # attention_mask: (B, T), where 1=real token, 0=pad
        if attention_mask is not None:
            # fix flash_attn with PyTorch back-end of scaled_dot_product_attention compatibility with is_causal=True
            # if self.flash_attn and torch.all(attention_mask == 1):
            # but make force solution more universally for both modes
            if torch.all(attention_mask == 1):
                # force fire-down attention mask for Flash Attention
                attn_mask = None
            else:
                # use attention_mask for manual attention implementation
                assert attention_mask.size(0) == B
                assert attention_mask.size(1) == T
                attn_mask = (attention_mask == 0)[:, None, None, :].to(device=x.device, dtype=torch.bool)  # (B,1,1,T)
        else:
            attn_mask = None


        if self.n_head == 1 or not self.flash_attn:
            # manual attention implementation
            attn = (q @ k.transpose(-2, -1)) / math.sqrt(k.size(-1)) # (B, nh, T, T)

            # causal mask: (1, 1, T, T)
            causal_mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), 1
                )[None, None, :, :]

            # combine both masks
            if attn_mask is not None:
                # invert: True = masked
                full_mask = causal_mask | (~attn_mask)
            else:
                full_mask = causal_mask

            # apply mask
            attn = attn.masked_fill(full_mask, float('-inf'))

            attn = F.softmax(attn, dim=-1)
            y = attn @ v  # (B, nh, T, hs)
        else:
            # use PyTorch flash attention (scaled_dot_product_attention)
            y = F.scaled_dot_product_attention(q, k, v,  attn_mask=attn_mask, is_causal=True) # flash attention

        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        hidden_dim = 4 * config.n_embd

        self.c_fc    = nn.Linear(config.n_embd, hidden_dim)
        self.gelu    = nn.GELU(approximate='tanh')
        self.c_proj  = nn.Linear(hidden_dim, config.n_embd)
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

    def forward(self, x, attention_mask=None):
        x = x + self.attn(self.ln_1(x), attention_mask=attention_mask)
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):

    def __init__(self, config=None, **kwargs):
        super().__init__()
        if config is None:
            config = GPTConfig(**kwargs)

        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight

        # init params
        self.apply(self._init_weights)


    def forward(self, idx, targets=None, attention_mask=None):
        # idx is of shape (B, T)
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        # forward the token and posisition embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape (T)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (T, n_embd)
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (B, T, n_embd)
        x = tok_emb + pos_emb
        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x, attention_mask=attention_mask)
        # forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)
        loss = None
        if targets is not None:
            if attention_mask is not None:
                # do not calculate paddings in loss
                targets = targets.clone()
                targets[attention_mask == 0] = -100
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100)
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


    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}

        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])
        return model


    def configure_optimizers(self, weight_decay, learning_rate, device_type, master_process):
        # start with all of the candidate parameters (that require grad)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        if master_process:
            print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
            print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        if master_process:
            print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer


    def get_num_params(self, non_embedding: bool = True, only_trainable: bool = False):

        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)

        if non_embedding:
            pe = self.transformer.wpe.weight.numel()
            total -= pe
        if only_trainable and self.transformer.wpe.weight.requires_grad:
            trainable -= pe

        value = trainable if only_trainable else total
        return value



    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,    # (B, T)
        attention_mask: Optional[torch.Tensor] = None,  # (B, T)
        max_new_tokens: int = 5,
        temperature: float = 1.0,
        do_sample: bool = False,
        top_k: int | None = None,
        eos_token_id: int | None = None,
        pad_token_id: int | None = None,
    ) -> torch.Tensor:

        self.eval()
        block_size = self.config.block_size

        if pad_token_id is None:
            pad_token_id = eos_token_id if eos_token_id is not None else 0

        if attention_mask is not None:
            B, T0 = input_ids.shape

            attention_mask = attention_mask.to(device=input_ids.device, dtype=torch.long)
            assert attention_mask.dim() == 2 and attention_mask.size(0) == B, f"bad attention_mask {attention_mask.shape}"
            assert attention_mask.size(1) == input_ids.size(1), f"mask/input mismatch: {attention_mask.size()} vs {input_ids.size()}"


        pad = torch.tensor(pad_token_id, device=input_ids.device, dtype=input_ids.dtype)
        finished = torch.zeros(input_ids.size(0), device=input_ids.device, dtype=torch.bool)

        for _ in range(max_new_tokens):
            if eos_token_id is not None and torch.all(finished):
                break

            idx_cond = input_ids if input_ids.size(1) <= block_size else input_ids[:, -block_size:]

            if attention_mask is None:
                logits = self(idx_cond).logits # (B, t, V)
            else:

                if input_ids.size(1) <= block_size:
                    am_cond = attention_mask
                else:
                    am_cond = attention_mask[:, -block_size:] if attention_mask is not None else None

                # strong correspondence in length to idx_cond
                assert am_cond.size(1) == idx_cond.size(1)
                logits = self(idx_cond, attention_mask=am_cond).logits  # (B, t, V)

            logits = logits[:, -1, :]       # (B, V)

            if do_sample:
                logits = logits / temperature
                if top_k is not None:
                    # clamp top_k to valid range [1, V]
                    V = logits.size(-1)
                    k = max(1, min(int(top_k), V))
                    v, _ = torch.topk(logits, k, dim=-1)
                    logits[logits < v[:, [-1]]] = -float("Inf")

                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)      # (B,1)
            else:
                idx_next = torch.argmax(logits, dim=-1, keepdim=True)   # (B,1)

            # if sequence already finished -> keep padding
            if eos_token_id is not None:
                idx_next = torch.where(finished[:, None], pad, idx_next)
                finished = finished | (idx_next.squeeze(1) == eos_token_id)

            input_ids = torch.cat((input_ids, idx_next), dim=1)
            if attention_mask is not None:
                next_mask = (~finished).to(dtype=attention_mask.dtype).unsqueeze(1)  # (B,1) 1=active, 0=finished
                attention_mask = torch.cat((attention_mask, next_mask), dim=1)

        return input_ids

