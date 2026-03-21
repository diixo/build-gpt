
import torch
from torch.nn import functional as F
import os, torch


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


def plot_loss(losses: list, model_type: str):
    import matplotlib.pyplot as plt

    plt.plot(range(1, len(losses) + 1), losses, marker='o', label="Training Loss")
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.title(f"Trained on: {model_type}")
    plt.legend()

    plt.grid(True, which="both", linestyle="--", alpha=0.5)  # <- грид
    plt.tight_layout()

    plt.show()


def create_hf_llama(tokenizer):

    from transformers import LlamaConfig, LlamaForCausalLM

    # "LLaMA-style" (RMSNorm + RoPE), by size is GPT-2 small:
    # n_layer=12, n_head=12, hidden=768, mlp=3072
    config = LlamaConfig(
        vocab_size=len(tokenizer),           # GPT-2
        hidden_size=768,            # n_embd
        intermediate_size=3072,     # usually 4 * hidden
        num_hidden_layers=12,       # n_layer
        num_attention_heads=12,     # n_head
        num_key_value_heads=12,     # = n_head -> regular MHA (without GQA)
        max_position_embeddings=1024,  # context-size (GPT-2 small = 1024)

        # typical llama-params
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        attention_bias=True,
        mlp_bias=False,
        tie_word_embeddings=True,

        # llama specified bos/eos; use default params
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        #bos_token_id=tokenizer.bos_token_id,
    )

    model = LlamaForCausalLM(config)
    model.config.use_cache = False

    params = sum(p.numel() for p in model.parameters())
    print("hf_llama.params:", params)
    # print("dtype:", next(model.parameters()).dtype)
    # print("device:", next(model.parameters()).device)
    return model


def create_hf_gpt2(tokenizer):

    from transformers import GPT2Config, GPT2LMHeadModel

    config = GPT2Config(
        vocab_size=len(tokenizer),
        n_positions=1024,
        n_embd=768,
        n_layer=12,
        n_head=12,
        n_inner=3072,
        activation_function="gelu_new",
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        attn_pdrop=0.1,
        layer_norm_epsilon=1e-5,
        initializer_range=0.02,
        tie_word_embeddings=True,

        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        #bos_token_id=tokenizer.bos_token_id,
    )
    #config.loss_type = "cross_entropy"

    model = GPT2LMHeadModel(config)
    model.config.use_cache = False
    return model


def file_path_from_config(model_type: str, train_config, save_directory):

    epochs = getattr(train_config, "epochs", "") or ""
    batch_size = getattr(train_config, "batch_size", "") or ""
    grad_accum_steps = getattr(train_config, "grad_accum_steps", "") or ""

    file_name = f"model_{model_type}-{epochs}-{batch_size}-{grad_accum_steps}.pt"

    return os.path.join(save_directory, file_name)


def save_trained_model(model_dir, model, model_type: str, train_config, tokenizer_type: str, **extra):

    os.makedirs(model_dir, exist_ok=True)

    ckpt = {
        "model": model.state_dict(),
        "config": (model.config if isinstance(model.config, dict) else getattr(model.config, "__dict__", None)),
        "train_config": (train_config if isinstance(train_config, dict) else getattr(train_config, "__dict__", None)),
        #"config": model.config,
        #"train_config": train_config,
        "tokenizer_type": tokenizer_type,
        "extra": extra,
    }
    # if optimizer is not None:
    #     ckpt["optimizer"] = optimizer.state_dict()

    checkpoint_path = file_path_from_config(model_type, train_config, model_dir)
    torch.save(ckpt, checkpoint_path)

