import json
import os
from pathlib import Path
import torch, math, random, numpy as np
from dataclasses import dataclass
from model_gpt2 import GPT, GPTNeo
from model_llama import GPTLlama
from model_gptx import GPTNeoX
from model_gpt_hybrid import GPTNeoHybrid
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from transformers import AutoTokenizer, GPT2Tokenizer, AutoModelForCausalLM, GPT2TokenizerFast
from transformers import set_seed


SEED = 42
set_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


class AutoGPTModel:

    MODEL_MAP = {
        "gpt": GPT,
        "gpt-neo": GPTNeo,
        "gpt-llama": GPTLlama,
        "gpt-neox": GPTNeoX,
        "gpt-neo-hybrid": GPTNeoHybrid,
    }

    CONFIG_MAP = {
        "gpt": dict(),
        "gpt-neo": dict(),
        "gpt-llama": dict(rope_base=10000.0, use_rope=True),
        "gpt-neox": dict(rope_base=10000.0, use_rope=True, rotary_pct=0.25, tie_word_embeddings=True),
        "gpt-neo-hybrid": dict(rope_base=10000.0, use_rope=True, rotary_pct=0.25, tie_word_embeddings=True),
    }

    @staticmethod
    def from_config(model_type: str):
        if model_type not in AutoGPTModel.MODEL_MAP:
            raise ValueError(f"Unknown model_type: {model_type}")

        #tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b", local_files_only=True)
        #tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-31m", local_files_only=True)
        tokenizer = GPT2Tokenizer.from_pretrained("data/gpt2", local_files_only=True)

        # Extract sizes
        vocab_sz = tokenizer.vocab_size # 50257
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
            "block_size": 1024,
            "vocab_size": vocab_sz,
            "n_layer": 8,
            "n_head": 12,
            "n_embd": 768,
            "flash_attn": True,
        })

        print(f"config_kwargs =\n{json.dumps(config_kwargs, indent=2)}")

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

    for item in batch:

        new_item = item.tolist() + [eos_token_id]

        # Pad sequences to max_length
        padded = new_item + [pad_token_id] * (batch_max_length - len(new_item))

        inputs = torch.tensor(padded[:-1])  # Truncate the last token for inputs
        targets = torch.tensor(padded[1:])  # Shift +1 to the right for targets

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

        inputs_lst.append(inputs)
        targets_lst.append(targets)

    # Convert list of inputs and targets to tensors and transfer to target device
    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)
    return inputs_tensor, targets_tensor


class TextDataset(Dataset):

    def __init__(self, file_path, tokenizer, max_seq_length=1000):

        texts = []
        with open(file_path, "r", encoding="utf-8") as f:
            texts = [line.strip() for line in f if line.strip()]

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


@dataclass
class TrainerConfig:
    epochs: int = 5
    batch_size: int = 8
    learning_rate: float = 1e-4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class Trainer:

    def __init__(self, model, dataset, config):
        self.losses = []

        self.model = model.to(config.device)
        self.config = config
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
        self.loader = DataLoader(
            dataset,
            batch_size = config.batch_size,
            shuffle=False,
            collate_fn=lambda batch: custom_collate_fn(
                batch,
                max_seq_length = model.config.block_size,
                pad_token_id = tokenizer.eos_token_id,
                eos_token_id = tokenizer.eos_token_id,
                device = config.device,
                ),
            )


    def train(self):

        grad_accum_steps = min(4, len(self.loader))

        self.losses = []

        self.model.train()
        for epoch in range(self.config.epochs):
            pbar = tqdm(self.loader, desc=f"Epoch {epoch + 1}/{self.config.epochs}")
            total_epoch_loss = 0.0
            smoothed_loss = 0.0

            self.optimizer.zero_grad(set_to_none=True)

            for step, (x, y) in enumerate(pbar):
                x, y = x.to(self.config.device), y.to(self.config.device)

                # Forward pass
                raw_loss = self.model(x, y).loss

                # Loss normalization for gradient accumulation
                loss = raw_loss / grad_accum_steps
                loss.backward()

                # Optimizer step
                if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(self.loader):
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)

                # Updating metrics
                iter_loss = raw_loss.item()
                total_epoch_loss += iter_loss

                # Exponential smoothing for progress-bar
                smoothed_loss = 0.9 * smoothed_loss + 0.1 * iter_loss if step > 0 else iter_loss
                pbar.set_postfix(loss=f"{smoothed_loss:.4f}")

            # Statistics at the end of the epoch
            avg_loss = total_epoch_loss / len(self.loader)
            self.losses.append(avg_loss)

            # Calculate Perplexity
            perplexity = math.exp(avg_loss) if avg_loss < 20 else float('inf')
            print(f"Epoch {epoch+1}: avg loss={avg_loss:.4f}, PPL={perplexity:.2f}")

        print("✅ Training completed.")


def test_collate_fn(pad_token_id, eos_token_id, ignore_index = -100):

    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    inputs_1 = torch.tensor([1, 2, 3, 4, 5],
                            dtype=torch.long)
    inputs_2 = torch.tensor([6, 7],
                            dtype=torch.long)
    inputs_3 = torch.tensor([7, 8, 9],
                            dtype=torch.long)

    batch = [inputs_1, inputs_2, inputs_3]

    print(64 * "*")
    inputs, targets = custom_collate_fn(
        batch,
        max_seq_length = 4,
        pad_token_id = pad_token_id,
        eos_token_id = eos_token_id,
        ignore_index = ignore_index,
        device = device)

    print(inputs)
    print(targets)


def _fmt(n: int) -> str:
    if n >= 1_000_000:
        return f"{n/1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n/1_000:.2f}K"
    return str(n)


if __name__ == "__main__":

    #test_collate_fn(pad_token_id=tokenizer.eos_token_id, eos_token_id=tokenizer.eos_token_id)

    #########################################################################################

    model, tokenizer = AutoGPTModel.from_config("gpt-llama")

    # Checking the types:
    print("Model type:", type(model))
    print("Tokenizer type:", type(tokenizer))

    config = TrainerConfig(epochs=5, batch_size=4)
    dataset = TextDataset("dataset.txt", tokenizer, max_seq_length=model.config.block_size)
    trainer = Trainer(model, dataset, config)
    trainer.train()

    input_ids = tokenizer("Model", truncation=True, add_special_tokens=False, return_tensors="pt")["input_ids"]
    gen_ids = model.generate(
                input_ids=input_ids.to(config.device),
                max_new_tokens=5,
                do_sample=True,
                top_k=10,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id
            )[0]
    output_text = tokenizer.decode(gen_ids, skip_special_tokens=True)

    print(f"Total model.params: {_fmt(model.get_num_params())}")
    print("Generated text:", output_text)

