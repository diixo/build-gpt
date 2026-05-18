import os
import re
from collections import Counter
import multiprocessing as mp
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq

import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import GPT2TokenizerFast


tokenizer = GPT2TokenizerFast.from_pretrained("data/gpt-noomo-32k", local_files_only=True)

MAX_LEN = 0
SPLIT_COUNTER = 0
WIKIPEDIA_PARQUET_DIR = Path("datasets/wikipedia/20220301.en")
# hf download aitetic/wikipedia --repo-type dataset --include "20220301.en/*" --local-dir ./datasets/wikipedia


# use for hugging-face tokenizer
def hf_tokenize_to_len(text):
    # tokenizes a single document and returns a numpy array of uint16 tokens
    ids = tokenizer.encode(text, add_special_tokens=False)
    if MAX_LEN > 0:
        if len(ids) > MAX_LEN:
            global SPLIT_COUNTER
            SPLIT_COUNTER += 1
        return min(len(ids), MAX_LEN)
    return len(ids)



def get_wikipedia_parquet_files():
    parquet_files = sorted(WIKIPEDIA_PARQUET_DIR.glob("*.parquet"))
    if not parquet_files:
        return []

    match = re.search(r"-of-(\d+)\.parquet$", parquet_files[0].name)
    if match:
        expected_parquet_files_count = int(match.group(1))
        if len(parquet_files) < expected_parquet_files_count:
            return []

    return parquet_files


def iter_wikipedia_texts(parquet_files, batch_size=1024):
    for parquet_file in parquet_files:
        parquet = pq.ParquetFile(parquet_file)
        for batch in parquet.iter_batches(batch_size=batch_size, columns=["text"]):
            yield from batch.column(0).to_pylist()


def count_wikipedia_rows(parquet_files):
    return sum(pq.ParquetFile(parquet_file).metadata.num_rows for parquet_file in parquet_files)


if __name__ == "__main__":

    parquet_files = get_wikipedia_parquet_files()
    if not parquet_files:
        raise SystemExit(0)

    parquet_files_count = len(parquet_files)

    total_docs = count_wikipedia_rows(parquet_files)

    size_map_counter = Counter()
    nprocs = min(4, os.cpu_count())

    total_tokens = 0

    with mp.Pool(nprocs) as pool:
        with tqdm(total=total_docs, unit="rows") as pbar:
            for token_len in pool.imap(hf_tokenize_to_len, iter_wikipedia_texts(parquet_files), chunksize=16):
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
    plt.xlim(0, 8200)
    plt.xlabel("Length of tokenized rows")
    plt.ylabel("Count of rows")
    plt.title(f"Tokens(all={total_tokens}), SPLIT_ITEMS={SPLIT_COUNTER}: Wikipedia 20220301.en ({parquet_files_count} parquet files)")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()
