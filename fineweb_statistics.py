import os
from collections import Counter
import multiprocessing as mp
import numpy as np
import tiktoken
import matplotlib.pyplot as plt
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer


#enc = tiktoken.get_encoding("gpt2")
#eot = enc._special_tokens['<|endoftext|>']  # end of text token


tokenizer = AutoTokenizer.from_pretrained("data/noomo", use_fast=True, local_files_only=True)
eot = tokenizer.convert_tokens_to_ids("<|endoftext|>")


# use for tiktoken tokenizer
# def tt_tokenize_to_len(row):
#     tokens = [eot]
#     tokens.extend(enc.encode_ordinary(row["text"]))
#     tokens_np = np.array(tokens, dtype=np.uint16)
#     #return tokens_np.astype(np.uint16)
#     return len(tokens_np)


# use for hugging-face tokenizer
def hf_tokenize_to_len(row):
    # tokenizes a single document and returns a numpy array of uint16 tokens
    tokens = [eot] # eot-token delimits all documents
    ids = tokenizer.encode(row["text"], add_special_tokens=False)
    tokens.extend(ids)
    tokens_np = np.array(tokens)
    assert (0 <= tokens_np).all() and (tokens_np < 2**16).all(), "token dictionary too large for uint16"
    return len(tokens_np)
    #return tokens_np.astype(np.uint16)


if __name__ == "__main__":

    fw = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train")

    total_docs = len(fw)  # 9_672_101

    size_map_counter = Counter()
    nprocs = min(4, os.cpu_count())

    total_tokens = 0

    with mp.Pool(nprocs) as pool:
        with tqdm(total=total_docs, unit="rows") as pbar:
            for token_len in pool.imap(hf_tokenize_to_len, fw, chunksize=16):
                size_map_counter[token_len] += 1
                total_tokens += token_len
                pbar.update(1)

    print(f"\nTokens.all={total_tokens}, nprocs={nprocs}")

    size_map_sorted = dict(sorted(size_map_counter.items(), key=lambda item: item[0], reverse=True))


    ###############################################################################################

    plt.figure(figsize=(9, 5))
    lengths = list(size_map_sorted.keys())
    counts = list(size_map_sorted.values())

    plt.hist(lengths, weights=counts, bins=200)
    plt.xlim(0, 8000)
    plt.xlabel("Length of tokenized rows")
    plt.ylabel("Count of rows")
    plt.title(f"Tokens(all={total_tokens}): FineWeb-Edu-10BT")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()
