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

* [fineweb.py](fineweb.py): build `edu_fineweb10B` dataset
* [main_ddp.py](main_ddp.py])
* [main.py](main.py)
* [test_models.py](test_models.py)


### Additional:

* train_tokenizer.py : train custom tokenizer
* fineweb_gpt.py     : build dataset from trained tokenizer
* fineweb_statistics.py : create tokens statistic from dataset

| Tokenizer             |     Tokens     |
|-----------------------|----------------|
| gpt-neo-125m          | 00000000000000 |
| noomo                 | 00000000000000 |
