
from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Any, Optional

import json
from transformers import AutoTokenizer, GPT2TokenizerFast


# -------------------------
# CONFIG
# -------------------------
DATA_PATH = "dataset.txt"
MY_TOK_PATH = "data/gpt-noomo-32k"
GPT2_TOK_PATH = "data/gpt2"
MAX_PRINT_TOKENS = 200

# -------------------------
# HELPERS
# -------------------------
def read_lines(path: str) -> List[str]:
    p = Path(path)
    lines = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.rstrip("\n")
            if s.strip() == "":
                continue
            lines.append(s)
    return lines


def safe_load_tokenizer(name_or_path: str):
    tok = AutoTokenizer.from_pretrained(name_or_path, use_fast=True, local_files_only=True)
    return tok


def get_tokens(tok, text: str) -> Dict[str, Any]:
    enc = tok(text, add_special_tokens=False, return_attention_mask=False, return_token_type_ids=False)
    ids = enc["input_ids"]
    tokens = tok.convert_ids_to_tokens(ids)

    # bytes per token (примерная метрика “насколько токены длинные”)
    bpt = (len(text.encode("utf-8")) / max(1, len(ids)))

    # “пробельный” токен-старт (под разные системы)
    space_like = 0
    for t in tokens:
        if t.startswith(("Ġ", "▁")) or t.startswith(" ") or t.startswith("Ċ") or t.startswith("▏"):
            space_like += 1
    unk_id = tok.unk_token_id
    unk_cnt = sum(1 for i in ids if (unk_id is not None and i == unk_id))
    return {
        "ids": ids,
        "tokens": tokens,
        "n": len(ids),
        "bytes_per_token": bpt,
        "space_like_ratio": space_like / max(1, len(tokens)),
        "unk_cnt": unk_cnt,
    }


def short_list(x: List[Any], k: int = MAX_PRINT_TOKENS) -> List[Any]:
    return x if len(x) <= k else (x[:k] + ["…"] + x[-10:])


def summarize(all_rows: List[Dict[str, Any]]) -> Dict[str, float]:
    # all_rows: list of per-line dicts
    total_tokens = sum(r["n"] for r in all_rows)
    total_unk = sum(r["unk_cnt"] for r in all_rows)
    avg_bpt = sum(r["bytes_per_token"] for r in all_rows) / max(1, len(all_rows))
    avg_space = sum(r["space_like_ratio"] for r in all_rows) / max(1, len(all_rows))
    return {
        "rows": float(len(all_rows)),
        "total_tokens": float(total_tokens),
        "avg_tokens_per_line": float(total_tokens / max(1, len(all_rows))),
        "avg_bytes_per_token": float(avg_bpt),
        "avg_space_like_ratio": float(avg_space),
        "total_unk": float(total_unk),
    }


def main():
    lines = read_lines(DATA_PATH)
    print(f"Loaded lines: {len(lines)} from {DATA_PATH}")

    tok_gpt2 = safe_load_tokenizer(GPT2_TOK_PATH)
    tok_my = safe_load_tokenizer(MY_TOK_PATH)

    gpt2_rows = []
    my_rows = []

    for i, text in enumerate(lines, 1):
        g = get_tokens(tok_gpt2, text)
        m = get_tokens(tok_my, text)
        gpt2_rows.append(g)
        my_rows.append(m)

        print("\n" + "="*80)
        print(f"LINE {i}: {text}")
        print("-"*80)
        print(f"GPT2: n={g['n']}  bpt={g['bytes_per_token']:.2f}  space_like={g['space_like_ratio']:.2f}  unk={g['unk_cnt']}")
        print("GPT2 tokens:", short_list(g["tokens"]))
        print("-"*80)
        print(f"MY  : n={m['n']}  bpt={m['bytes_per_token']:.2f}  space_like={m['space_like_ratio']:.2f}  unk={m['unk_cnt']}")
        print("MY tokens  :", short_list(m["tokens"]))

        # “где сильнее отличается”
        if abs(g["n"] - m["n"]) >= 10:
            print(f"NOTE: token count differs by {m['n'] - g['n']} tokens on this line.")

    print("\n" + "#"*80)
    print("SUMMARY")
    print("GPT2:", json.dumps(summarize(gpt2_rows), ensure_ascii=False, indent=2))
    print("MY  :", json.dumps(summarize(my_rows), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
