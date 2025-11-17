from collections import Counter
import multiprocessing as mp
import numpy as np
import json
import tiktoken
import matplotlib.pyplot as plt
from tqdm import tqdm
from datasets import load_dataset


enc = tiktoken.get_encoding("gpt2")
eot = enc._special_tokens['<|endoftext|>']  # end of text token


def tokenize_to_len(doc):
    tokens = [eot]
    tokens.extend(enc.encode_ordinary(doc["text"]))
    tokens_np = np.array(tokens, dtype=np.uint16)
    return len(tokens_np)


if __name__ == "__main__":

    fw = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train")

    total_docs = len(fw)  # 9_672_101

    size_map_counter = Counter()
    nprocs = 4  # max(1, os.cpu_count() // 2)

    with mp.Pool(nprocs) as pool:
        with tqdm(total=total_docs, unit='docs') as pbar:
            for token_len in pool.imap(tokenize_to_len, fw, chunksize=16):
                size_map_counter[token_len] += 1
                pbar.update(1)

    size_map_sorted = dict(sorted(size_map_counter.items(), key=lambda item: item[0], reverse=True))

    # сохраняем результат в JSON
    with open("size-map.json", "w", encoding="utf-8") as f:
        json.dump(size_map_sorted, f, ensure_ascii=False, indent=4)

    total = sum(size_map_counter.values())


    ##################################################

    # сортируем по длине токенов для красивого графика
    lengths = sorted(size_map_counter.keys())
    counts = [size_map_counter[l] for l in lengths]

    plt.figure(figsize=(10, 6))
    plt.bar(lengths, counts, color='skyblue')
    plt.xlabel("Length of tokenized rows")
    plt.ylabel("Count of rows")
    plt.title("Sizes distribution of tokenized rows from dataset FineWeb-Edu-10BT")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()
