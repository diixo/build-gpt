import json
import os
import re
from pathlib import Path
import torch, math, random, numpy as np
import pyarrow.parquet as pq
from dataclasses import dataclass
from itertools import islice
from model_llama import GPTLlama
from torch.utils.data import Dataset, DataLoader, IterableDataset
from tqdm import tqdm

from transformers import GPT2TokenizerFast
from transformers import set_seed

import matplotlib.pyplot as plt


WIKIPEDIA_PARQUET_DIR = Path("datasets/wikipedia_20220301_en/data/20220301.en")
# hf download legacy-datasets/wikipedia --repo-type dataset --include "data/20220301.en/*" --local-dir ./datasets/wikipedia_20220301_en

DEFAULT_SMOKE_ROWS = 608
TRAIN_MODE = "smoke-train"
SMOKE_ROWS = DEFAULT_SMOKE_ROWS

MAX_LEN = 1024
BLOCK_SIZE = 4096


@dataclass
class TrainerConfig:
    epochs: int = 1
    batch_size: int = 4
    learning_rate: float = 8e-5
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    grad_accum_steps: int = 1


class AutoConfigModel:

    MODEL_MAP = {
        "gpt2",
        "mini",
    }

    @staticmethod
    def from_config(size_type: str, tokenizer_type="gpt2"):

        if size_type not in AutoConfigModel.MODEL_MAP:
            raise ValueError(f"Unknown size_type: {size_type}")

        tokenizer = GPT2TokenizerFast.from_pretrained(f"data/{tokenizer_type}", local_files_only=True)

        # Extract sizes
        vocab_sz = len(tokenizer.get_vocab())   # size include special tokens
        print("Vocab size: tokenizer =", vocab_sz)


        # Check alls special tokens
        print(f"Special tokens =", tokenizer.special_tokens_map)

        # Check eos_token_id and the token itself
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print("EOS token string:", repr(tokenizer.convert_ids_to_tokens(tokenizer.eos_token_id)))


        config_kwargs = dict(rope_base=10000.0, use_rope=True)

        if size_type == "gpt2":
            config_kwargs.update({
                "block_size": BLOCK_SIZE,
                "vocab_size": vocab_sz,
                "n_layer": 12,
                "n_head": 12,
                "n_embd": 768,
                "flash_attn": True,
                "model_type": size_type,
            })

        elif size_type == "mini":
            config_kwargs.update({
                "block_size": BLOCK_SIZE,
                "vocab_size": vocab_sz,
                "n_layer": 16,
                "n_head": 16,
                "n_embd": 1024,
                "flash_attn": True,
                "model_type": size_type,
            })

        print(f"config_kwargs =\n{json.dumps(config_kwargs, indent=2)}")

        # get the model class
        model = GPTLlama(**config_kwargs)

        return model, tokenizer


def custom_collate_fn(batch, max_seq_length, pad_token_id, eos_token_id, device, ignore_index=-100):
    """
    Custom collate function for variable-length text samples.

    Args:
        batch: list of tokenized samples
        eos_token_id: int, used for padding termination
        device: torch.device

    Returns:
        inputs_tensor: [batch_size, seq_len]
        targets_tensor: [batch_size, seq_len]
        attention_mask: [batch_size, seq_len]
    """

    # Find the longest sequence in the batch
    batch_max_length = max(len(item) + 1 for item in batch)

    # Pad and prepare inputs and targets
    inputs_lst, targets_lst = [], []
    attn_lst = []

    for item in batch:

        new_item = item.tolist() + [eos_token_id]
        real_len = len(new_item)

        # Pad sequences to max_length
        padded = new_item + [pad_token_id] * (batch_max_length - real_len)

        # build attention mask from real_len (NOT from token values)
        attn = [1] * real_len + [0] * (batch_max_length - real_len)

        inputs = torch.tensor(padded[:-1])
        targets = torch.tensor(padded[1:])
        am = torch.tensor(attn[:-1], dtype=torch.long)

        # Replace all but the first padding tokens in targets by ignore_index
        mask = targets == pad_token_id
        indices = torch.nonzero(mask).squeeze()
        if indices.numel() > 1:
            targets[indices[1:]] = ignore_index

        if max_seq_length is not None:
            inputs = inputs[:max_seq_length]
            targets = targets[:max_seq_length]
            am = am[:max_seq_length]

        inputs_lst.append(inputs)
        targets_lst.append(targets)
        attn_lst.append(am)

    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)
    attention_mask = torch.stack(attn_lst).to(device)
    return inputs_tensor, targets_tensor, attention_mask



def get_wikipedia_parquet_files(parquet_dir=WIKIPEDIA_PARQUET_DIR):
    parquet_files = sorted(Path(parquet_dir).glob("*.parquet"))
    if not parquet_files:
        return []

    match = re.search(r"-of-(\d+)\.parquet$", parquet_files[0].name)
    if match:
        expected_parquet_files_count = int(match.group(1))
        if len(parquet_files) < expected_parquet_files_count:
            return []

    return parquet_files


def count_wikipedia_rows(parquet_files):
    return sum(pq.ParquetFile(parquet_file).metadata.num_rows for parquet_file in parquet_files)


class WikipediaParquetDataset(IterableDataset):

    def __init__(self, tokenizer, max_seq_length=MAX_LEN-1, parquet_dir=WIKIPEDIA_PARQUET_DIR, batch_size=1024, max_rows=None):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size
        self.max_rows = max_rows
        self.parquet_files = get_wikipedia_parquet_files(parquet_dir)
        total_rows = count_wikipedia_rows(self.parquet_files) if self.parquet_files else 0
        self.total_rows = min(total_rows, max_rows) if max_rows is not None else total_rows

        print(
            f"WikipediaParquetDataset::loaded files.sz={len(self.parquet_files)}, rows.sz={self.total_rows}, max_rows={self.max_rows}"
        )

    def __len__(self):
        return self.total_rows

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        parquet_files = self.parquet_files
        if worker_info is not None:
            parquet_files = parquet_files[worker_info.id::worker_info.num_workers]

        texts_iter = self._iter_texts(parquet_files)
        if self.max_rows is not None:
            texts_iter = islice(texts_iter, self.max_rows)

        for text in texts_iter:
            yield self.tokenizer(
                text,
                truncation=True,
                add_special_tokens=False,
                max_length=self.max_seq_length,
                padding=False,
                return_tensors="pt",
            )["input_ids"].squeeze(0)

    def _iter_texts(self, parquet_files):
        for parquet_file in parquet_files:
            parquet = pq.ParquetFile(parquet_file)
            for batch in parquet.iter_batches(batch_size=self.batch_size, columns=["text"]):
                for text in batch.column(0).to_pylist():
                    if text is None:
                        yield ""
                    elif isinstance(text, str):
                        yield text
                    else:
                        yield str(text)


class Trainer:

    def __init__(self, model, dataset, config):
        self.losses = []

        self.model = model.to(config.device).float()
        self.config = config
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
        self.loader = DataLoader(
            dataset,
            batch_size = config.batch_size,
            shuffle=False,
            collate_fn=lambda batch: custom_collate_fn(
                batch,
                #max_seq_length = model.config.block_size,
                max_seq_length = MAX_LEN,
                pad_token_id = tokenizer.eos_token_id,
                eos_token_id = tokenizer.eos_token_id,
                device = config.device,
                ),
            )


    def train(self):

        torch.set_float32_matmul_precision("high")

        # 1) Gradient accumulation should be an explicit hyperparameter
        grad_accum_steps = int(getattr(self.config, "grad_accum_steps", 1))
        grad_accum_steps = max(1, grad_accum_steps)
        # if epoch has fewer batches than accum steps — clamp
        grad_accum_steps = min(grad_accum_steps, max(1, len(self.loader)))

        self.losses = []          # token-weighted epoch losses (good for PPL)
        self.step_losses = []     # avg per-window accumulation raw loss (for plotting)

        self.model.train()
        for epoch in range(self.config.epochs):
            pbar = tqdm(self.loader, desc=f"Epoch {epoch + 1}/{self.config.epochs}")

            total_loss_sum = 0.0   # sum of (mean_loss * num_valid_tokens)
            total_tokens = 0       # number of non-ignored tokens
            first_loss = None

            self.optimizer.zero_grad(set_to_none=True)

            accum_raw_sum = 0.0

            for step, batch in enumerate(pbar):
                # NOTE: your collate already .to(device), so these .to() are redundant but harmless
                #x = x.to(self.config.device, non_blocking=True)
                #y = y.to(self.config.device, non_blocking=True)

                if isinstance(batch, dict):
                    input_ids = batch["input_ids"]
                    labels = batch["labels"]
                    attention_mask = batch.get("attention_mask", None)
                else:
                    if len(batch) == 2:
                        input_ids, labels = batch
                        attention_mask = None
                    else:
                        input_ids, labels, attention_mask = batch


                # Forward pass
                raw_loss = self.model(input_ids, labels).loss

                # ---- logging helpers ----
                raw = float(raw_loss.detach().cpu().item())
                accum_raw_sum += raw

                # token-weighted stats for correct epoch avg loss / PPL
                with torch.no_grad():
                    ntok = int((labels != -100).sum().item())
                total_loss_sum += raw * ntok
                total_tokens += ntok

                # ---- backward (accumulation) ----
                loss = raw_loss / grad_accum_steps
                loss.backward()

                # Progress bar smoothing
                if first_loss is None:
                    first_loss = raw
                    pbar.set_postfix(loss=f"{first_loss:.4f}", accum_steps=str(grad_accum_steps))


                # Optimizer step
                if ((step + 1) % grad_accum_steps == 0) or ((step + 1) == len(self.loader)):
                    if getattr(self.config, "max_grad_norm", None) is not None:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(self.config.max_grad_norm))
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)

                    # calculate the average raw loss for current accumulation window
                    step_avg_loss = accum_raw_sum / grad_accum_steps
                    accum_raw_sum = 0.0
                    self.step_losses.append(step_avg_loss)

                    pbar.set_postfix(loss=f"{step_avg_loss:.4f}", accum_steps=str(grad_accum_steps))


            # ---- epoch metrics (token-weighted, correct for variable lengths) ----
            if total_tokens == 0:
                epoch_avg_loss = float("nan")
                ppl = float("nan")
            else:
                epoch_avg_loss = total_loss_sum / total_tokens
                # # Calculate Perplexity, avoid overflow for huge losses
                ppl = math.exp(epoch_avg_loss) if epoch_avg_loss < 50 else float("inf")

            self.losses.append(epoch_avg_loss)
            print(f"Epoch {epoch+1}: epoch_avg_loss={epoch_avg_loss:.4f}, PPL={ppl:.4f}")

        print(
            "✅ Training completed,",
            f"steps: {len(self.step_losses)}, final_avg_loss: {self.losses[-1]:.4f}"
        )

        return self.losses, self.step_losses


def plot_losses(losses1: list, label1: str, x_label: str):

    plt.plot(range(len(losses1)), losses1, label=label1, color="blue")

    plt.xlabel(x_label)
    plt.ylabel("Loss")
    plt.title(f"Training")
    plt.legend()

    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    tokenizer_type = "gpt-noomo-32k"

    model: GPTLlama = None

    train_config = TrainerConfig(epochs=1, batch_size=1, grad_accum_steps=4)

    model, tokenizer = AutoConfigModel.from_config(size_type="mini", tokenizer_type=tokenizer_type)

    smoke_rows = SMOKE_ROWS if TRAIN_MODE == "smoke-train" else None
    print(f"model.sz={model.get_num_params()}, train_mode={TRAIN_MODE}, smoke_rows={smoke_rows}")

    dataset = WikipediaParquetDataset(tokenizer, max_seq_length=MAX_LEN - 1, max_rows=smoke_rows)
    if len(dataset) == 0:
        raise SystemExit(0)

    trainer = Trainer(model, dataset, train_config)

    epoch_losses, step_losses = trainer.train()

    plot_losses(step_losses, type(model).__name__, "Steps")
