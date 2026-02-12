
import json
from pathlib import Path
from transformers import GPT2TokenizerFast

path = Path("data/dictionary.cambridge.org-505600.jsonl")
tok = GPT2TokenizerFast.from_pretrained("gpt2")

max_len = -1
total_len = 0
max_id = None
max_text = None
n = 0
bad = 0


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
            continue

        text = obj.get("example")
        if not isinstance(text, str) or not text:
            continue

        n += 1
        length = len(tok.encode(text))
        total_len += length
        if length > max_len:
            max_len = length
            max_id = i
            max_text = text

print(f"Records used: {n}, bad json lines: {bad}")
print(f"MAX tokens: {max_len}, id={max_id}, total_tokens={total_len}")

#print("Winner preview:", (max_text[:300] + "…") if max_text and len(max_text) > 300 else max_text)
