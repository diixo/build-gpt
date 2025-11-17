
# count_token_freqs.py
from datasets import load_dataset
from transformers import AutoTokenizer
from collections import Counter
from tqdm import tqdm
import math

DATASET = "HuggingFaceFW/fineweb-edu"
NAME = "sample-10BT"
SPLIT = "train"
TOKENIZER_PATH = "noomo"   # уже сохранённый hf tokenizer dir (PreTrainedTokenizerFast)
BATCH = 512                # ставь меньше, если мало RAM

def batch_generator(ds, batch_size=BATCH):
    batch = []
    for ex in ds:
        txt = ex.get("text") or ex.get("content") or ex.get("body")
        if not txt:
            continue
        if isinstance(txt, list):
            for t in txt:
                if t:
                    batch.append(t)
        else:
            batch.append(txt)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch

def main():
    ds = load_dataset(DATASET, name=NAME, split=SPLIT)
    total = len(ds)
    print("Total docs:", total)

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, use_fast=True)
    freq = Counter()
    docs = 0
    with tqdm(total=total, unit="docs") as pbar:
        for batch in batch_generator(ds):
            enc = tokenizer(batch, add_special_tokens=False)
            for ids in enc["input_ids"]:
                freq.update(ids)
            docs += len(batch)
            pbar.update(len(batch))

    # summary
    total_tokens = sum(freq.values())
    vocab_size = len(freq)
    print("Total token occurrences:", total_tokens)
    print("Unique token ids seen:", vocab_size)

    # distribution buckets
    buckets = [(1,1), (2,4), (5,9), (10,49), (50,199), (200,999), (1000, math.inf)]
    for low, high in buckets:
        c = sum(1 for t,f in freq.items() if low <= f <= (high if math.isfinite(high) else 10**18))
        print(f"tokens with freq in [{low}, {high}]: {c}")

    # top-30 tokens (show token string too)
    inv_vocab = {v:k for k,v in tokenizer.get_vocab().items()}
    top30 = freq.most_common(30)
    print("Top 30 tokens (id, freq, token):")
    for tid, f in top30:
        tok = inv_vocab.get(tid, "<UNK_ID>")
        print(tid, f, tok)

if __name__ == "__main__":
    main()
