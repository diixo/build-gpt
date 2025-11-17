from collections import Counter
import multiprocessing as mp
import numpy as np
import json
import tiktoken
import matplotlib.pyplot as plt
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer


enc = tiktoken.get_encoding("gpt2")
# eot = enc._special_tokens['<|endoftext|>']  # end of text token


tokenizer = AutoTokenizer.from_pretrained("data/gpt-neo-125m", use_fast=True)
eot = tokenizer.convert_tokens_to_ids("<|endoftext|>")


def tiktoken_to_len(row):
    tokens = [eot]
    tokens.extend(enc.encode_ordinary(row["text"]))
    tokens_np = np.array(tokens, dtype=np.uint16)
    #return tokens_np.astype(np.uint16)
    return len(tokens_np)


def tokenize_to_len(row):
    # tokenizes a single document and returns a numpy array of uint16 tokens
    tokens = [eot] # the special <|endoftext|> token delimits all documents
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
    nprocs = 4  # max(1, os.cpu_count() // 2)

    with mp.Pool(nprocs) as pool:
        with tqdm(total=total_docs, unit='rows') as pbar:
            for token_len in pool.imap(tokenize_to_len, fw, chunksize=16):
                size_map_counter[token_len] += 1
                pbar.update(1)

    size_map_sorted = dict(sorted(size_map_counter.items(), key=lambda item: item[0], reverse=True))

    # save results into JSON:
    #with open("size-map.json", "w", encoding="utf-8") as f:
    #    json.dump(size_map_sorted, f, ensure_ascii=False, indent=4)

    total = sum(size_map_counter.values())

    print(f"\nTokens.all={total}")


    ###############################################################################################

    lengths = sorted(size_map_counter.keys())
    filtered_lengths = [l for l in lengths if l < 16]
    counts = [size_map_counter[l] for l in filtered_lengths]

    plt.figure(figsize=(9, 5))
    plt.bar(filtered_lengths, counts, color='skyblue')
    plt.xlabel("Length of tokenized rows")
    plt.ylabel("Count of rows")
    plt.title(f"Tokens(all={total}) sizes distribution tokenization of dataset: FineWeb-Edu-10BT")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()
