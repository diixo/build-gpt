
import math
from typing import Optional, Dict, Any, Iterable

import torch
import torch.nn.functional as F
from datasets import load_dataset

import os
from tqdm import tqdm
import torch
from torch.nn import functional as F
from transformers import GPT2TokenizerFast, AutoModelForCausalLM
from model_gpt2 import GPT, GPTConfig


torch.set_float32_matmul_precision('high') # use tf32

# -----------------------------------------------------------------------------
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), "hellaswag")

device = "cuda" if torch.cuda.is_available() else "cpu"

import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset


class WikiText103PerplexityDataset(Dataset):

    def __init__(
        self,
        tokenizer,
        split: str = "test",
        block_size: int = 1024,
        stride: int = None,
    ):
        self.block_size = block_size
        self.stride = stride if stride is not None else block_size
        self.examples = []

        token_buffer = []

        ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split=split)
        pbar = tqdm(ds, desc=f"Tokenizing WikiText-103, part=\"{split}\"")

        for item in pbar:
            text = item["text"].strip()
            if not text:
                continue

            ids = tokenizer.encode(text, add_special_tokens=False)

            if len(ids) == 0:
                continue

            # добавляем разделитель между кусками текста
            token_buffer.extend(ids)

            # режем готовые блоки из буфера
            while len(token_buffer) >= block_size:
                chunk = token_buffer[:block_size]
                self.examples.append(torch.tensor(chunk, dtype=torch.long))

                if self.stride == block_size:
                    token_buffer = token_buffer[block_size:]
                else:
                    token_buffer = token_buffer[self.stride:]


    def __len__(self):
        return len(self.examples)


    def __getitem__(self, idx):
        input_ids = self.examples[idx]
        attention_mask = torch.ones_like(input_ids)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }


def get_wikitext103_perplexity_loader(
    tokenizer,
    split: str = "test",
    block_size: int = 1024,
    batch_size: int = 8,
    stride: int = None,
    num_workers: int = 0,
    pin_memory: bool = True,
):
    dataset = WikiText103PerplexityDataset(
        tokenizer=tokenizer,
        split=split,
        block_size=block_size,
        stride=stride,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return loader


@torch.no_grad()
def evaluate_perplexity(
    model,
    dataloader: Iterable[Dict[str, Any]],
    device: torch.device,
    pad_token_id: Optional[int] = None,
    max_batches: Optional[int] = None,
    use_attention_mask: bool = True,
    show_progress: bool = True,
) -> Dict[str, float]:
    """
    Computes perplexity for a causal language model (decoder-only, nanoGPT-style).

    Expected batch format:
        {
            "input_ids": LongTensor [B, T],
            optionally "attention_mask": LongTensor [B, T]
        }

    The function computes standard next-token loss:
        predict x[:, 1:] from x[:, :-1]

    Args:
        model:
            Causal LM model. Must return logits of shape [B, T, V]
            either directly, or in an object with `.logits`.
        dataloader:
            Iterable of batches.
        device:
            torch.device("cuda") or torch.device("cpu")
        pad_token_id:
            If provided, tokens equal to this id in shifted labels are ignored.
        max_batches:
            If provided, evaluate only this many batches.
        use_attention_mask:
            Whether to pass attention_mask to model when present in batch.

    Returns:
        dict with:
            - avg_nll: average negative log-likelihood per predicted token
            - perplexity: exp(avg_nll)
            - total_loss_tokens: number of tokens used in loss
    """

    model.eval()

    total_nll = 0.0
    total_loss_tokens = 0

    iterator = dataloader
    if show_progress:
        total = max_batches if max_batches is not None else None
        iterator = tqdm(dataloader, total=total, desc="Evaluating PPL")

    for batch_idx, batch in enumerate(iterator):
        if max_batches is not None and batch_idx >= max_batches:
            print("<<<<")
            break

        input_ids = batch["input_ids"].to(device)  # [B, T]
        attention_mask = batch.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        if input_ids.size(1) < 2:
            continue

        # Forward
        if use_attention_mask and attention_mask is not None:
            outputs = model(input_ids, attention_mask=attention_mask)
        else:
            outputs = model(input_ids)

        if hasattr(outputs, "logits"):
            logits = outputs.logits
        elif isinstance(outputs, tuple):
            logits = outputs[0]
        else:
            logits = outputs

        # Shift for next-token prediction
        shift_logits = logits[:, :-1, :].contiguous()   # [B, T-1, V]
        shift_labels = input_ids[:, 1:].contiguous()    # [B, T-1]

        # Build valid-token mask
        valid_mask = torch.ones_like(shift_labels, dtype=torch.bool)

        if pad_token_id is not None:
            valid_mask &= (shift_labels != pad_token_id)

        if attention_mask is not None:
            # label at position t is predicted from token at t-1,
            # so validity should correspond to label position
            shift_attn = attention_mask[:, 1:].contiguous().bool()
            valid_mask &= shift_attn

        # Flatten
        vocab_size = shift_logits.size(-1)
        flat_logits = shift_logits.view(-1, vocab_size)
        flat_labels = shift_labels.view(-1)
        flat_mask = valid_mask.view(-1)

        valid_tokens = flat_mask.sum().item()
        if valid_tokens == 0:
            continue

        # Per-token NLL
        per_token_nll = F.cross_entropy(
            flat_logits,
            flat_labels,
            reduction="none"
        )  # [B*(T-1)]

        per_token_nll = per_token_nll[flat_mask]

        total_nll += per_token_nll.sum().item()
        total_loss_tokens += valid_tokens

        if show_progress and hasattr(iterator, "set_postfix"):
            avg_nll = total_nll / total_loss_tokens
            ppl = math.exp(avg_nll) if avg_nll < 20 else float("inf")
            iterator.set_postfix({
                "avg_nll": f"{avg_nll:.4f}",
                "ppl": f"{ppl:.2f}",
            })

    if total_loss_tokens == 0:
        raise ValueError("No valid tokens were found for perplexity computation.")

    avg_nll = total_nll / total_loss_tokens
    perplexity = math.exp(avg_nll) if avg_nll < 20 else float("inf")

    return {
        "avg_nll": avg_nll,
        "perplexity": perplexity,
        "total_loss_tokens": total_loss_tokens,
    }


def load_saved_model():

    file_path = "models/nano-gpt/model_19072.pt"

    config = GPTConfig(block_size=1024, vocab_size=50304, n_layer=12, n_head=12, n_embd=768)

    ckpt = torch.load(file_path, map_location=device, weights_only=False)

    config = ckpt['config']

    model = GPT(**config) if isinstance(config, dict) else GPT(config)

    model.load_state_dict(ckpt['model'])
    return model


if __name__ == "__main__":

    #model = load_saved_model()
    model = AutoModelForCausalLM.from_pretrained("gpt2")

    model.to(device)
    model.eval()

    ##################################################################################

    tokenizer = GPT2TokenizerFast.from_pretrained(f"data/gpt2", local_files_only=True)

    test_loader = get_wikitext103_perplexity_loader(
        tokenizer=tokenizer,
        split="train",
        block_size=1024,
        batch_size=8,
    )

    # raw version: текст почти как в оригинале
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")

    def nonempty_texts(split):
        return [x["text"] for x in ds[split] if x["text"].strip()]

    train_ds = nonempty_texts("train")
    val_ds   = nonempty_texts("validation")
    test_ds  = nonempty_texts("test")

    #print(train_ds[:3])

    #print(train_ds[0])
    print(len(train_ds), len(val_ds), len(test_ds))

    metrics = evaluate_perplexity(
        model=model,
        dataloader=test_loader,
        device=device,
        pad_token_id=tokenizer.eos_token_id,   # или tokenizer.pad_token_id
        max_batches=None,
        use_attention_mask=True,
    )

    print(metrics)
    print("PPL:", metrics["perplexity"])