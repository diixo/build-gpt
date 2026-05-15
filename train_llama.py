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


MAX_LEN = 4096


class AutoGPTModel:

    MODEL_MAP = {
        "gpt2": GPT,
        "llama": GPTLlama,
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

        # get the model class
        model_cls = AutoGPTModel.MODEL_MAP[model_type]
        model = model_cls(**config_kwargs)

        return model, tokenizer


if __name__ == "__main__":

    set_seed(42)

    model_type = "llama"
    tokenizer_type = "gpt-noomo-32k"

    model, tokenizer = AutoGPTModel.from_config(model_type=model_type, tokenizer_type=tokenizer_type)

    print(f"Number of parameters: {model.get_num_params()}")
