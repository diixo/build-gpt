import json
from model_gpt2 import GPT, GPTNeo
from model_llama import GPTLlama
from model_gptx import GPTNeoX

from transformers import AutoTokenizer


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
        tokenizer = AutoTokenizer.from_pretrained("gpt2")

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



if __name__ == "__main__":

    model, tokenizer = AutoGPT2Model.from_config("gpt")

    # Checking the types:
    print("Model type:", type(model))
    print("Tokenizer type:", type(tokenizer))
