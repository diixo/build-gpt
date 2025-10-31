# build-gpt

Based on: https://github.com/karpathy/build-nanogpt

* **GPT**: standard nanoGPT.
* **GPTNeo**: GPT model with static sin-cos embeddings, that extended positional embeddings (as in GPT-Neo).
* **GPTLlama**: GPT model with simple RoPE.
* **GPTNeoX**: GPT model, featured: percentage RoPE, QK-normalization, RMSNormFn.
* **GPTNeoHybrid**: GPT model (based on GPTNeoX), featured: percentage RoPE, QK-normalization, RMSNormFn.

## TODO:
Integrate `attention_mask` from GPT-model to another models.


### Demos:

* main_ddp.py
* main.py
* test_models.py
