# build-gpt

* Based on: https://github.com/karpathy/build-nanogpt

A PyTorch library with educational re-implementation of GPT-models: LLaMA, [GPT2](https://github.com/openai/gpt-2), GPTNeo, GPTNeoX, included both training and inference. Code tries to be small, clean and interpretable, as most of the currently available GPT model implementations can a bit sprawling. Code focused on implementation of [GPTLlama](model_llama.py), that is not a complicated model and this implementation is appropriately about 350 lines of code.


### Demos:

* [fineweb.py](fineweb.py): build tokenized `edu_fineweb10B` dataset.
* [main_ddp.py](main_ddp.py]): train model in `ddp` mode.
* [main.py](main.py): train model in `single-GPU` mode.


### Implemented models:

* [**GPT-model**](model_gpt2.py): standard nanoGPT model.
* [**GPTNeo**](model_gpt2.py): GPT model with static sin-cos embeddings, that extended positional embeddings (as in GPT-Neo).
* [**GPTLlama**](model_llama.py): GPT model with simple RoPE, RMSNorm-trainable, SWiGLU.
* [**GPTNeoX**](model_gptx.py): GPT model, featured: percentage RoPE, QK-normalization, RMSNormFn.
* [**GPTNeoHybrid**](model_gpt_hybrid.py): GPT model (based on GPTNeoX), featured: percentage RoPE, QK-normalization, RMSNormFn.


### Additionally:

* [train_tokenizer.py](train_tokenizer.py)      : train custom tokenizer
* [fineweb_gpt.py](fineweb_gpt.py)              : build dataset from trained tokenizer
* [fineweb_statistics.py](fineweb_statistics.py): create tokens statistic from dataset


### Testing models:

```python
MODEL_MAP = {
    "gpt": GPT,
    "gpt-neo": GPTNeo,
    "gpt-llama": GPTLlama,
    "gpt-neox": GPTNeoX,
    "gpt-neo-hybrid": GPTNeoHybrid,
}
```

[test_models.py](test_models.py): testing models functionality (**gpt-llama** current model).


### Train tokenizer:

* [train_tokenizer.py](train_tokenizer.py): train custom tokenizer
* [compare_tokenizers.py](compare_tokenizers.py): comparing tokenizers by quality
* [fineweb_statistics.py](fineweb_statistics.py): tokens statistic of dataset **Fineweb-Edu-10BT**

| Tokenizer             |     Tokens      |   size   |
|-----------------------|-----------------|----------|
| gpt2                  |  9_953_989_344  |  50_257  |
| gpt-neo-125m          |  9_953_989_333  |  50_257  |
| pythia-31m            |  9_919_456_391  |  50_304  |
| noomo                 |  9_843_922_538  |  40_257  |

