
import json
from pathlib import Path
from transformers import GPT2TokenizerFast


#tokenizer = GPT2TokenizerFast.from_pretrained("data/noomo", local_files_only=True)
tokenizer = GPT2TokenizerFast.from_pretrained("data/gpt-noomo-32k", local_files_only=True)

paths = [
    "data/dictionary.cambridge.org-00.jsonl",
    "data/dictionary.cambridge.org-01.jsonl",
    ]


max_len = -1
total_len = 0
max_id = None
max_text = None
n = 0
bad = 0


for path_str in paths:
    path = Path(path_str)

    with path.open("r", encoding="utf-8") as f:

        i = 0
        while True:
            line = f.readline()
            if not line:  # EOF
                break

            i += 1
            if i % 1000 == 0:
                print(f"...{i} lines, current max_tokens: {max_len}, total_tokens: {total_len}, bad lines: {bad}")

            try:
                obj = json.loads(line)
            except Exception:
                bad += 1
                print(f"Bad line at {i}: {line[:32]}…")
                continue

            text = obj.get("example")
            if not isinstance(text, str) or not text:
                continue

            n += 1
            length = len(tokenizer.encode(text))
            total_len += length
            if length > max_len:
                max_len = length
                max_id = i
                max_text = text

print(f"Items: {n}, bad json lines: {bad}")
print(f"MAX_tokens: {max_len}, line_id={max_id}, total_tokens={total_len}")

print("### Max-tokens line preview:", (max_text[:100] + "…") if max_text and len(max_text) > 100 else max_text)
