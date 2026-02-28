from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.trainers import BpeTrainer
from transformers import PreTrainedTokenizerFast, GPT2TokenizerFast
from tqdm import tqdm
import json
from datasets import Dataset, concatenate_datasets


EOT = "<|endoftext|>"
tokenizer_path = "data/noomo"

def read_jsonl(file_path: str) -> list:
    text = []
    with open(file_path, "r", encoding="utf-8") as f:

        for line in f:
            item = json.loads(line)
            if item.get("example"):
                text.append(item["example"])
            else:
                title = item["title"]
                description = item["description"]
                text.append(title + " " + description)
    return text


fw = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train")

fw_extended = concatenate_datasets([
    #Dataset.from_dict({ "text": read_jsonl("data/dictionary.cambridge.org-00.jsonl") }),
    #Dataset.from_dict({ "text": read_jsonl("data/dictionary.cambridge.org-01.jsonl") }),
    fw,
    Dataset.from_dict({ "text": read_jsonl("datasets/arxiv-corpus/arxiv_cs_2015_2020.jsonl") }),
    Dataset.from_dict({ "text": read_jsonl("datasets/arxiv-corpus/arxiv_cs_2021_2024.jsonl") }),
    ])

total_rows = len(fw_extended)

print(f"total_rows: {total_rows}")  # 11104126

################################################################################################

def text_iterator():
    for row in tqdm(fw_extended, total=total_rows, unit="rows"):
        txt = row.get("text", "")
        if txt and isinstance(txt, str):
            yield txt

################################################################################################

tokenizer = Tokenizer(BPE())
tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=True)
tokenizer.decoder = ByteLevelDecoder()

trainer = BpeTrainer(
    vocab_size=40_256,
    min_frequency=5,
    initial_alphabet=ByteLevel.alphabet(),
    special_tokens=[]
)

tokenizer.train_from_iterator(text_iterator(), trainer=trainer)

tokenizer.add_special_tokens([EOT])


fast_tokenizer = PreTrainedTokenizerFast(
    tokenizer_object = tokenizer,
    eos_token = EOT,
    bos_token = None,
    unk_token = None
)

fast_tokenizer.save_pretrained(tokenizer_path)
