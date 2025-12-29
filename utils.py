
import torch
from torch.nn import functional as F


def generate_text(prompt: str, model, enc, device, device_type, ddp_rank):
    model.eval()
    num_return_sequences = 1
    max_length = 64
    tokens = enc.encode(prompt)
    tokens = torch.tensor(tokens, dtype=torch.long)
    tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
    xgen = tokens.to(device)
    sample_rng = torch.Generator(device=device)
    sample_rng.manual_seed(42 + ddp_rank)

    with torch.no_grad():
        while xgen.size(1) < max_length:
            # forward the model to get the logits
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                logits = model(xgen).logits # (B, T, vocab_size)
            # take the logits at the last position
            logits = logits[:, -1, :] # (B, vocab_size)
            # get the probabilities
            probs = F.softmax(logits, dim=-1)
            # do top-k sampling of 50 (huggingface pipeline default)
            # topk_probs here becomes (5, 50), topk_indices is (5, 50)
            topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
            # select a token from the top-k probabilities
            # note: multinomial does not demand the input to sum to 1
            ix = torch.multinomial(topk_probs, 1, generator=sample_rng) # (B, 1)
            # gather the corresponding indices
            xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
            # append to the sequence
            xgen = torch.cat((xgen, xcol), dim=1)
    # print the generated text
    for i in range(num_return_sequences):
        tokens = xgen[i, :max_length].tolist()
        decoded = enc.decode(tokens)
        print(f"rank {ddp_rank} sample {i}: {decoded}")


def plot_loss(losses: list):
    import matplotlib.pyplot as plt

    plt.plot(losses, label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss over Epoch")
    plt.legend()
    plt.show()


########################################################################
from dataclasses import dataclass
from typing import Optional, List, Union
import torch
import torch.nn.functional as F


def _apply_top_k(logits: torch.Tensor, k: int) -> torch.Tensor:
    if k <= 0 or k >= logits.size(-1):
        return logits
    vals, _ = torch.topk(logits, k, dim=-1)
    kth = vals[:, -1].unsqueeze(-1)
    return logits.masked_fill(logits < kth, float("-inf"))

def _apply_top_p(logits: torch.Tensor, p: float) -> torch.Tensor:
    if p >= 1.0:
        return logits
    sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
    probs = F.softmax(sorted_logits, dim=-1)
    cum = torch.cumsum(probs, dim=-1)

    mask = cum > p
    mask[:, 0] = False  # keep at least 1 token
    sorted_logits = sorted_logits.masked_fill(mask, float("-inf"))

    unsorted = torch.empty_like(sorted_logits)
    unsorted.scatter_(dim=-1, index=sorted_idx, src=sorted_logits)
    return unsorted

def _repetition_penalty_(next_logits: torch.Tensor, seq: torch.Tensor, cur_len: int, penalty: float):
    # in-place modify next_logits for one sample
    if penalty == 1.0:
        return
    seen = seq[:cur_len]
    for tok in torch.unique(seen):
        tok = int(tok.item())
        s = next_logits[tok]
        next_logits[tok] = s / penalty if s > 0 else s * penalty


@dataclass
class GenerateOutput:
    sequences: torch.Tensor
    scores: Optional[List[torch.Tensor]] = None  # list of (B, vocab)


def generate_gpt(
    model,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    max_new_tokens: int = 20,
    max_length: Optional[int] = None,
    do_sample: bool = False,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    repetition_penalty: float = 1.0,
    eos_token_id: Optional[int] = None,
    pad_token_id: Optional[int] = None,
    num_return_sequences: int = 1,
    return_dict_in_generate: bool = False,
    output_scores: bool = False,
) -> Union[torch.Tensor, GenerateOutput]:
    """
    HF-like generate() as a standalone function (no beams, no KV-cache).
    Expects model.forward(input_ids, attention_mask=...) returning an object with `.logits` of shape (B,T,V).
    attention_mask is expected as (B,T) with 1 for real tokens, 0 for padding (right padding).
    """
    if input_ids.dim() != 2:
        raise ValueError(f"input_ids must be (B,T), got {tuple(input_ids.shape)}")
    if attention_mask is not None and attention_mask.shape != input_ids.shape:
        raise ValueError(
            f"attention_mask must match input_ids shape, got {tuple(attention_mask.shape)} vs {tuple(input_ids.shape)}"
        )

    if temperature <= 0:
        raise ValueError("temperature must be > 0")

    device = input_ids.device
    input_ids = input_ids.long()

    # defaults
    if eos_token_id is None:
        eos_token_id = getattr(getattr(model, "config", None), "eos_token_id", None)
    if pad_token_id is None:
        pad_token_id = eos_token_id if eos_token_id is not None else 0

    model.eval()

    scores: Optional[List[torch.Tensor]] = [] if output_scores else None

    with torch.inference_mode():
        # expand for num_return_sequences
        B, T = input_ids.shape

        if num_return_sequences > 1:
            input_ids = input_ids.repeat_interleave(num_return_sequences, dim=0)
            if attention_mask is not None:
                attention_mask = attention_mask.repeat_interleave(num_return_sequences, dim=0)
            B = input_ids.size(0)

        # prompt lengths (supports right padding)
        if attention_mask is None:
            prompt_lens = torch.full((B,), T, device=device, dtype=torch.long)
        else:
            prompt_lens = attention_mask.long().sum(dim=1)
            prompt_lens = torch.clamp(prompt_lens, min=1)

        # compute new tokens count
        if max_length is not None:
            max_new_tokens = max(0, int(max_length) - int(prompt_lens.max().item()))
        max_new_tokens = int(max_new_tokens)

        # decide total output length (cap by model context)
        block_size = getattr(getattr(model, "config", None), "block_size", None)
        if block_size is None:
            raise ValueError("model.config.block_size is required for left-truncation/capping")

        cur_max_len = int(prompt_lens.max().item())
        total_len = min(block_size, cur_max_len + max_new_tokens)

        # output buffer
        sequences = torch.full((B, total_len), int(pad_token_id), device=device, dtype=torch.long)
        copy_len = min(T, total_len)
        sequences[:, :copy_len] = input_ids[:, :copy_len]

        cur_lens = prompt_lens.clamp(max=total_len)
        finished = torch.zeros((B,), device=device, dtype=torch.bool)

        # how many steps can we do given total_len cap
        max_steps = max(0, total_len - int(cur_lens.max().item()))

        for _ in range(max_steps):
            if torch.all(finished):
                break

            t_cur_max = int(cur_lens.max().item())

            # feed last window (left truncation)
            if t_cur_max <= block_size:
                model_input = sequences[:, :t_cur_max]
                if attention_mask is None:
                    model_attn = None
                else:
                    ar = torch.arange(t_cur_max, device=device).unsqueeze(0)
                    model_attn = (ar < cur_lens.unsqueeze(1)).long()
            else:
                start = t_cur_max - block_size
                model_input = sequences[:, start:t_cur_max]
                ar = torch.arange(block_size, device=device).unsqueeze(0)
                window_lens = (cur_lens - start).clamp(min=1, max=block_size)
                model_attn = (ar < window_lens.unsqueeze(1)).long()

            out = model(model_input, attention_mask=model_attn)
            logits = out.logits  # (B, t, V)

            # pick logits at last real token in window per sample
            if model_attn is None:
                last_pos = torch.full((B,), model_input.size(1) - 1, device=device, dtype=torch.long)
            else:
                last_pos = (model_attn.long().sum(dim=1) - 1).clamp(min=0)

            next_logits = logits[torch.arange(B, device=device), last_pos, :]  # (B,V)

            # repetition penalty (HF-like)
            if repetition_penalty is not None and repetition_penalty != 1.0:
                for b in range(B):
                    if finished[b]:
                        continue
                    _repetition_penalty_(next_logits[b], sequences[b], int(cur_lens[b].item()), float(repetition_penalty))

            # sampling controls
            if do_sample and temperature != 1.0:
                next_logits = next_logits / float(temperature)

            next_logits = _apply_top_k(next_logits, int(top_k))
            next_logits = _apply_top_p(next_logits, float(top_p))

            if output_scores:
                scores.append(next_logits.detach())

            # choose token
            if do_sample:
                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_token = torch.argmax(next_logits, dim=-1)

            # write token for unfinished
            for b in range(B):
                if finished[b]:
                    continue
                pos = int(cur_lens[b].item())
                if pos >= total_len:
                    finished[b] = True
                    continue
                sequences[b, pos] = next_token[b]
                cur_lens[b] = cur_lens[b] + 1

                if eos_token_id is not None and int(next_token[b].item()) == int(eos_token_id):
                    finished[b] = True

        # cut to max length actually written (keep padding inside that cut)
        final_len = int(cur_lens.max().item())
        final_len = max(1, min(final_len, sequences.size(1)))
        sequences = sequences[:, :final_len]

        if return_dict_in_generate:
            return GenerateOutput(sequences=sequences, scores=scores)
        return sequences

