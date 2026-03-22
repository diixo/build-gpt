import json

import torch, math, numpy as np
from dataclasses import dataclass
from model_gpt2 import GPT
from model_llama import GPTLlama
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from transformers import GPT2TokenizerFast

from utils import create_hf_llama, create_hf_gpt2

import matplotlib.pyplot as plt


MAX_LEN = 1024


@dataclass
class TrainerConfig:
    epochs: int = 1
    batch_size: int = 4
    learning_rate: float = 1e-4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    grad_accum_steps: int = 1


class AutoGPTModel:

    MODEL_MAP = {
        "gpt2": GPT,
        "llama": GPTLlama,
    }

    MODEL_FACTORY_MAP = {
        "gpt2": create_hf_gpt2,
        "llama": create_hf_llama,
    }

    CONFIG_MAP = {
        "gpt2": dict(),
        "llama": dict(rope_base=10000.0, use_rope=True),
    }


    @staticmethod
    def from_config(model_type: str, use_hf=False, tokenizer_type="gpt2"):

        if model_type not in AutoGPTModel.MODEL_MAP:
            raise ValueError(f"Unknown model_type: {model_type}")

        tokenizer = GPT2TokenizerFast.from_pretrained(f"data/{tokenizer_type}", local_files_only=True)

        # Extract sizes
        vocab_sz = len(tokenizer.get_vocab())   # size include special tokens
        print("Vocab size: tokenizer =", vocab_sz)

        if True:
            # Check alls special tokens
            print(f"Special tokens =", tokenizer.special_tokens_map)

            # Check eos_token_id and the token itself
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            print("EOS token string:", repr(tokenizer.convert_ids_to_tokens(tokenizer.eos_token_id)))


        config_kwargs = AutoGPTModel.CONFIG_MAP[model_type]

        config_kwargs.update({
            "block_size": MAX_LEN,
            "vocab_size": vocab_sz,
            "n_layer": 12,
            "n_head": 12,
            "n_embd": 768,
            "flash_attn": True,
            "model_type": model_type,
        })

        print(f"config_kwargs =\n{json.dumps(config_kwargs, indent=2)}")

        if use_hf:
            if model_type not in AutoGPTModel.MODEL_FACTORY_MAP:
                raise ValueError(f"Unknown model_type: {model_type}")

            model = AutoGPTModel.MODEL_FACTORY_MAP[model_type](tokenizer)
        else:
            # get the model class
            model_cls = AutoGPTModel.MODEL_MAP[model_type]
            model = model_cls(**config_kwargs)

        return model, tokenizer


def custom_collate_fn(batch, max_seq_length, pad_token_id, eos_token_id, device, ignore_index=-100):
    """
    Custom collate function for variable-length text samples.

    Args:
        batch: list of tuples (input_ids, target_ids)
        eos_token_id: int, used for padding
        device: torch.device
        allow_max_length: optional int, if set — limit max length of batch sequences

    Returns:
        inputs_tensor: [batch_size, seq_len]
        targets_tensor: [batch_size, seq_len]
    """

    # Find the longest sequence in the batch
    batch_max_length = max(len(item)+1 for item in batch)

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

        inputs = torch.tensor(padded[:-1])  # Truncate the last token for inputs
        targets = torch.tensor(padded[1:])  # Shift +1 to the right for targets
        am  = torch.tensor(attn[:-1], dtype=torch.long)

        # New: Replace all but the first padding tokens in targets by ignore_index
        mask = targets == pad_token_id
        # removes dimensions of size 1.
        indices = torch.nonzero(mask).squeeze()
        if indices.numel() > 1:
            targets[indices[1:]] = ignore_index

        # New: Optionally truncate to maximum sequence length
        if max_seq_length is not None:
            inputs = inputs[:max_seq_length]
            targets = targets[:max_seq_length]
            am  = am[:max_seq_length]

        inputs_lst.append(inputs)
        targets_lst.append(targets)
        attn_lst.append(am)


    # Convert list of inputs and targets to tensors and transfer to target device
    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)
    attention_mask = torch.stack(attn_lst).to(device)
    return inputs_tensor, targets_tensor, attention_mask


class TextDataset(Dataset):

    def __init__(self, file_path, tokenizer, max_seq_length=MAX_LEN-1):

        texts = []
        with open(file_path, "r", encoding="utf-8") as f:
            texts = [line.strip() for line in f if line.strip()]

        print(f"TextDataset::loaded items.sz={len(texts)}")

        # tokenize each line separately and store the input_ids, with only truncation, without padding
        self.data_idx = [
            tokenizer(t, truncation=True, add_special_tokens=False, max_length=max_seq_length, padding=False, return_tensors="pt"
            )["input_ids"].squeeze(0)   # sizes: [seq_len <= max_seq_length]
            for t in texts
        ]
        self.max_seq_length = max_seq_length


    def __len__(self):
        return len(self.data_idx)

    def __getitem__(self, idx):
        return self.data_idx[idx]



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


    # train model with hf-interface
    def train_hf(self):

        torch.set_float32_matmul_precision("high")

        grad_accum_steps = int(getattr(self.config, "grad_accum_steps", 1))
        grad_accum_steps = max(1, grad_accum_steps)
        grad_accum_steps = min(grad_accum_steps, max(1, len(self.loader)))

        self.losses = []       # token-weighted epoch losses (good for PPL)
        self.step_losses = []  # avg per-window accumulation raw loss (for plotting)

        self.model.train()

        for epoch in range(self.config.epochs):
            pbar = tqdm(self.loader, desc=f"Epoch {epoch + 1}/{self.config.epochs}")

            total_loss_sum = 0.0  # sum of (mean_loss * num_valid_tokens)
            total_tokens = 0      # number of non-ignored tokens
            first_loss = None

            self.optimizer.zero_grad(set_to_none=True)

            accum_raw_sum = 0.0

            for step, batch in enumerate(pbar):

                # ---- unpack batch ----
                # supports:
                # 1) (input_ids, labels)
                # 2) (input_ids, labels, attention_mask)
                # 3) dict: {"input_ids":..., "labels":..., "attention_mask":...}
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

                # Forward pass (HF LLaMA / AutoModelForCausalLM)
                out = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    use_cache=False,
                )
                raw_loss = out.loss

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

                if first_loss is None:
                    first_loss = raw
                    pbar.set_postfix(loss=f"{first_loss:.4f}", accum_steps=str(grad_accum_steps))

                # Optimizer step
                if ((step + 1) % grad_accum_steps == 0) or ((step + 1) == len(self.loader)):
                    if getattr(self.config, "max_grad_norm", None) is not None:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(self.config.max_grad_norm))

                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)

                    step_avg_loss = accum_raw_sum / grad_accum_steps
                    accum_raw_sum = 0.0
                    self.step_losses.append(step_avg_loss)

                    pbar.set_postfix(loss=f"{step_avg_loss:.4f}", accum_steps=str(grad_accum_steps))

            # ---- epoch metrics (token-weighted) ----
            if total_tokens == 0:
                epoch_avg_loss = float("nan")
                ppl = float("nan")
            else:
                epoch_avg_loss = total_loss_sum / total_tokens
                ppl = math.exp(epoch_avg_loss) if epoch_avg_loss < 50 else float("inf")

            self.losses.append(epoch_avg_loss)
            print(f"Epoch {epoch+1}: epoch_avg_loss={epoch_avg_loss:.4f}, PPL={ppl:.4f}")

        print(
            "✅ Training completed,",
            f"steps: {len(self.step_losses)}, final_avg_loss: {self.losses[-1]:.4f}"
        )

        return self.losses, self.step_losses


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

        print("✅ Training completed,",
            f"steps: {len(self.step_losses)}, final_avg_loss: {self.losses[-1]:.4f}")

        return self.losses, self.step_losses


def plot_losses(losses1: list, label1: str, losses2: list, label2: str, x_label: str):

    plt.plot(range(len(losses1)), losses1, label=label1, color="blue")
    plt.plot(range(len(losses2)), losses2, label=label2, color="red")

    plt.xlabel(x_label)
    plt.ylabel("Loss")
    plt.title(f"Training")
    plt.legend()

    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    model_type = "llama"
    tokenizer_type = "gpt-noomo-32k"    # "gpt2"


    train_config = TrainerConfig(epochs=25, batch_size=32, grad_accum_steps=1)

    ################################################################################################################
    model, tokenizer = AutoGPTModel.from_config(model_type=model_type, use_hf=False, tokenizer_type=tokenizer_type)

    model_hf, tokenizer = AutoGPTModel.from_config(model_type=model_type, use_hf=True, tokenizer_type=tokenizer_type)
    ################################################################################################################

    dataset = TextDataset("dataset.txt", tokenizer, max_seq_length=MAX_LEN)

    ################################################################################################################

    train_config.learning_rate = 1e-4
    trainer_hf = Trainer(model_hf, dataset, train_config)

    train_config.learning_rate = 8e-5
    trainer = Trainer(model, dataset, train_config)

    ################################################################################################################

    epoch_losses_hf, step_losses_hf = trainer_hf.train_hf()

    epoch_losses, step_losses = trainer.train()

    ################################################################################################################

    plot_losses(epoch_losses_hf, type(model_hf), epoch_losses, type(model), "Epochs")

    plot_losses(step_losses_hf, type(model_hf), step_losses, type(model), "Steps")
