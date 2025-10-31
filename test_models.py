import json
import os
import torch
from dataclasses import dataclass
from model_gpt2 import GPT, GPTNeo
from model_llama import GPTLlama
from model_gptx import GPTNeoX
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
#from functools import partial

from transformers import AutoTokenizer, GPT2Tokenizer


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

        new_item = item.copy() + [eos_token_id]

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

    def __init__(self, file_path, tokenizer, max_seq_length=1024):

        texts = []
        if isinstance(file_path, str) and os.path.isfile(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                texts = [line.strip() for line in f if line.strip()]

        # tokenize every line separately and store the input_ids
        encodings = [
            tokenizer(t.strip(), truncation=True, max_length=max_seq_length, padding="max_length", return_tensors="pt"
            )["input_ids"][0] for t in texts]

        self.data = torch.stack(encodings)
        self.max_seq_length = max_seq_length


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = x.clone()
        return x, y


@dataclass
class TrainerConfig:
    epochs: int = 5
    batch_size: int = 8
    learning_rate: float = 3e-4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class Trainer:

    def __init__(self, model, dataset, config):
        self.model = model.to(config.device)
        self.config = config
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
        self.loader = DataLoader(
            dataset,
            batch_size = config.batch_size,
            shuffle=False,
            collate_fn=lambda b: custom_collate_fn(
                b,
                max_seq_length=config.block_size,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                device=config.device,
                ),
            )


    def train(self):
        self.model.train()
        for epoch in range(self.config.epochs):
            pbar = tqdm(self.loader, desc=f"Epoch {epoch + 1} / {self.config.epochs}")
            for x, y in pbar:
                x, y = x.to(self.config.device), y.to(self.config.device)
                logits, loss = self.model(x, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                pbar.set_postfix(loss=loss.item())
        print("✅ Training complete.")


class AutoGPT2Model:

    MODEL_MAP = {
        "gpt": GPT,
        "gpt-neo": GPTNeo,
        "gpt-llama": GPTLlama,
        "gpt-neox": GPTNeoX,
    }

    CONFIG_MAP = {
        "gpt": dict(),
        "gpt-neo": dict(),
        "gpt-llama": dict(rope_base=10000.0, use_rope=True),
        "gpt-neox": dict(rope_base=10000.0, use_rope=True, rotary_pct=0.25),
    }

    @staticmethod
    def from_config(model_type: str):
        if model_type not in AutoGPT2Model.MODEL_MAP:
            raise ValueError(f"Unknown model_type: {model_type}")

        # For any model we use the same tokenizer (GPT-NeoX tokenizer)
        #tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        config_kwargs = AutoGPT2Model.CONFIG_MAP[model_type]
        config_kwargs.update({
            "block_size": 2048,
            "vocab_size": 50304,
            "n_layer": 12,
            "n_head": 12,
            "n_embd": 768,
            "flash_attn": True,
        })

        print(f"config_kwargs =\n{json.dumps(config_kwargs, indent=2)}")

        model_cls = AutoGPT2Model.MODEL_MAP[model_type]
        model = model_cls(**config_kwargs)
        return model, tokenizer


def test_collate_fn():

    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    inputs_1 = [0, 1, 2, 3, 4]
    inputs_2 = [5, 6]
    inputs_3 = [7, 8 , 9]

    batch = [inputs_1, inputs_2, inputs_3]

    print(24 * "*")
    inputs, targets = custom_collate_fn(
        batch,
        max_seq_length = 4,
        pad_token_id = 50256,
        eos_token_id = 50256,
        ignore_index = -100,
        device = device)

    print(inputs)
    print(targets)
    exit(0)


if __name__ == "__main__":

    #test_collate_fn()

    model, tokenizer = AutoGPT2Model.from_config("gpt")

    # Checking the types:
    print("Model type:", type(model))
    print("Tokenizer type:", type(tokenizer))

    dataset = TextDataset("test.txt", tokenizer, max_seq_length=model.config.block_size)
    trainer = Trainer(model, dataset, TrainerConfig(epochs=2, batch_size=4))
    exit(0)
    trainer.train()
