import json
import os
from pathlib import Path
import torch, math, random, numpy as np
from dataclasses import dataclass
from model_gpt2 import GPT
from model_llama import GPTLlama
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from transformers import GPT2TokenizerFast
from transformers import set_seed
from utils import create_hf_llama, create_hf_gpt2, plot_loss, save_trained_model, file_path_from_config


MAX_LEN = 1024
SAVE_DIRECTORY = "train_product"

USE_HF = False

SEED = 42
set_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


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
    def from_config(model_type: str, tokenizer_type="gpt2"):

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

        if USE_HF:
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


class JsonlDataset(Dataset):

    def __init__(self, file_pairs: list[tuple[str, str]], tokenizer, max_seq_length=1024):
        """
        files_map: list[(str, str)] # [(field_name: path_to_jsonl), ...]
        data_idx: list[Tensor]      # flat list of token id tensors
        """

        if not isinstance(file_pairs, list) or not file_pairs:
            raise ValueError("file_pairs must be a non-empty list of (field_name, path_to_jsonl) tuples")

        self.max_seq_length = max_seq_length
        self.data_idx = []

        total_texts = 0

        for field_name, file_path in file_pairs:
            texts = []
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    t = obj.get(field_name)
                    if t is None:
                        continue
                    if not isinstance(t, str):
                        t = str(t)
                    texts.append(t)

            print(f"JsonlDataset::field='{field_name}' loaded items.sz={len(texts)} from {file_path}")
            total_texts += len(texts)

            self.data_idx.extend(
                tokenizer(
                    texts,
                    truncation=True,
                    add_special_tokens=False,
                    max_length=max_seq_length,
                    padding=False,
                    return_tensors=None,
                )["input_ids"]  # list[list[int]]
            )

        print(f"JsonlDataset::TOTAL loaded items.sz={total_texts}")

        # convert each id-list to 1D LongTensor (as it was before)
        # (if tokenizer returned torch.Tensor already — should handle that too, just squeeze the extra dim)
        fixed = []
        for x in self.data_idx:
            if isinstance(x, torch.Tensor):
                fixed.append(x.squeeze(0))
            else:
                # always use when self.data_idx = list[list[int]]
                fixed.append(torch.tensor(x, dtype=torch.long))
        self.data_idx = fixed


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
            f"params: {_fmt(self.model.num_parameters()) if hasattr(self.model, 'num_parameters') else _fmt(self.model.get_num_params())}, "
            f"steps: {len(self.step_losses)}, final_avg_loss: {self.losses[-1]:.4f}"
        )

        return self.step_losses


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
            f"params: {_fmt(self.model.get_num_params())}, steps: {len(self.step_losses)}, final_avg_loss: {self.losses[-1]:.4f}")

        return self.step_losses


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


def load_pretrained_model(model_type: str, file_path: str, default_tokenizer_type="gpt2"):
    if model_type not in AutoGPTModel.MODEL_MAP:
        raise ValueError(f"Unknown model_type: {model_type}")

    if not os.path.exists(file_path): return None, None

    ckpt = torch.load(file_path, map_location="cpu", weights_only=False)

    extra = ckpt.get("extra", {})
    print("extra:", extra)

    config = ckpt['config']
    tokenizer_type = ckpt.get("tokenizer_type", default_tokenizer_type)

    tokenizer = GPT2TokenizerFast.from_pretrained(f"data/{tokenizer_type}", local_files_only=True)

    # get the model class from mapping
    model_cls = AutoGPTModel.MODEL_MAP[model_type]

    # create the model instance use mapped class
    model = model_cls(**config) if isinstance(config, dict) else model_cls(config)
    model.load_state_dict(ckpt['model'])

    return model, tokenizer


if __name__ == "__main__":

    #test_collate_fn(pad_token_id=0, eos_token_id=0, ignore_index=-100)

    #########################################################################################
    USE_TEST = True

    model_type = "llama"         # "gpt2", "llama"
    tokenizer_type = "gpt-noomo-32k"    # "gpt2", "gpt-noomo"


    train_config = TrainerConfig(epochs=25, batch_size=32, grad_accum_steps=1)

    model, tokenizer = load_pretrained_model(model_type, file_path_from_config(model_type, train_config, SAVE_DIRECTORY))

    if model is not None:
        model.to(train_config.device)
        print(f"✅ Loaded: model.total_params: {_fmt(model.get_num_params())}")
    else:
        model, tokenizer = AutoGPTModel.from_config(model_type, tokenizer_type)

        print("tokenizer.eos_token_id:", tokenizer.eos_token_id)

        if USE_TEST:
            dataset = TextDataset("test.txt", tokenizer, max_seq_length=MAX_LEN)
        else:
            dataset = JsonlDataset(
                [
                    ("example", "data/dictionary.cambridge.org-00.jsonl"),
                    ("example", "data/dictionary.cambridge.org-01.jsonl")
                ],
                tokenizer,
                max_seq_length=model.config.block_size
            )



        trainer = Trainer(model, dataset, train_config)

        if USE_HF:
            step_losses = trainer.train_hf()
        else:
            step_losses = trainer.train()


        avg_loss=round(float(trainer.losses[-1]), 4)

        extra = {"avg_loss": avg_loss, "examples_count": len(dataset)}

        if not USE_HF:
            save_trained_model(SAVE_DIRECTORY,
                                model,
                                model_type=model_type,
                                train_config=train_config,
                                tokenizer_type=tokenizer_type,
                                **extra)

        plot_loss(step_losses, type(model))


    # Checking the types:
    print(80 * "-")
    print("Model_type:", type(model))
    print("Tokenizer_type:", type(tokenizer))

    enc = tokenizer("Transformer", truncation=True, max_length=MAX_LEN, add_special_tokens=False, return_tensors="pt")

    input_ids = enc["input_ids"]

    if USE_TEST:
        attention_mask = enc["attention_mask"]
    else:
        attention_mask = None

    gen_ids = model.generate(
                input_ids=input_ids.to(train_config.device),
                attention_mask=attention_mask.to(train_config.device),
                max_new_tokens=5,
                do_sample=False,
                top_k=10,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id
            )[0]
    output_text = tokenizer.decode(gen_ids, skip_special_tokens=True)

    print("Generated text:", output_text)
