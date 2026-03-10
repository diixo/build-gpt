
from transformers import GPT2TokenizerFast, GPT2LMHeadModel


save_dir = "data/gpt2-extended"
model_name = "gpt2"


tokenizer = GPT2TokenizerFast.from_pretrained(save_dir)
model = GPT2LMHeadModel.from_pretrained(model_name)


'''
USER=QUESTION
ASSISTANT=ANSWER
KNOWLEDGE=CONTEXT
'''
special_tokens = {
    "pad_token": "<|pad|>",
    "additional_special_tokens": [
        "<|system|>",
        "<|user|>",
        "<|assistant|>",
        "<|knowledge|>",
        "<|instruction|>",
        "<|reference|>",
    ]
}

num_added = tokenizer.add_special_tokens(special_tokens)
print("added:", num_added)
print("vocab size:", len(tokenizer))

tokenizer.save_pretrained(save_dir)

#model.resize_token_embeddings(len(tokenizer))

#model.config.pad_token_id = tokenizer.pad_token_id
#model.save_pretrained(save_dir)
