# build-gpt

Based on: https://github.com/karpathy/build-nanogpt


### Demos:

* [fineweb.py](fineweb.py): build tokenized `edu_fineweb10B` dataset.
* [main_ddp.py](main_ddp.py]): train model in `ddp` mode.
* [main.py](main.py): train model in `single-GPU` mode.


### Additional:

* [train_tokenizer.py](train_tokenizer.py)      : train custom tokenizer
* [fineweb_gpt.py](fineweb_gpt.py)              : build dataset from trained tokenizer
* [fineweb_statistics.py](fineweb_statistics.py): create tokens statistic from dataset


### Custom models:

* [**GPT-model**](model_gpt2.py): standard nanoGPT model.
* [**GPTNeo**](model_gpt2.py): GPT model with static sin-cos embeddings, that extended positional embeddings (as in GPT-Neo).
* [**GPTLlama**](model_llama.py): GPT model with simple RoPE, RMSNorm-trainable, SWiGLU.
* [**GPTNeoX**](model_gptx.py): GPT model, featured: percentage RoPE, QK-normalization, RMSNormFn.
* [**GPTNeoHybrid**](model_gpt_hybrid.py): GPT model (based on GPTNeoX), featured: percentage RoPE, QK-normalization, RMSNormFn.


### Testing models

[test_models.py](test_models.py): testing models functionality (**gpt-llama** current model).


### Train tokenizer:

[train_tokenizer.py](train_tokenizer.py): train custom tokenizer

| Tokenizer             |     Tokens      |
|-----------------------|-----------------|
| gpt2                  |  9_953_989_344  |
| gpt-neo-125m          |  9_953_989_333  |
| pythia-31m            |  9_919_456_391  |
| noomo                 |  9_707_208_407  |


#### TODO:

* Integrate `attention_mask` from GPT-model to another models.


