
import torch
from torch.nn import functional as F
import torch.nn as nn


# class BlockPA(nn.Module):   # Block Parallel Attention
#     # head_layer+1 ​= head_layer ​+ MLP( LN(head_layer ​+ Attn(LN(head_layer))) )

#     def __init__(self, config):
#         super().__init__()
#         self.ln_attn = nn.LayerNorm(config.n_embd)
#         self.ln_mlp = nn.LayerNorm(config.n_embd)
#         self.attn = CausalSelfAttention(config)
#         self.mlp = MLP(config)

#     def forward(self, x):
#         x = x + self.attn(self.ln_attn(x)) + self.mlp(self.ln_mlp(x))
#         return x


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

        while xgen.size(1) < max_length:
            # forward the model to get the logits
            with torch.no_grad():
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
