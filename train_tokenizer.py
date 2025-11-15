from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.trainers import BpeTrainer
from transformers import PreTrainedTokenizerFast, GPT2TokenizerFast
from tqdm import tqdm


EOT = "<|endoftext|>"
tokenizer_path = "noomo"


fw = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train")
total_docs = len(fw)


tokenizer = Tokenizer(BPE())
tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=True)
tokenizer.decoder = ByteLevelDecoder()

trainer = BpeTrainer(
    vocab_size=50_256,
    min_frequency=2,
    initial_alphabet=ByteLevel.alphabet(),
    special_tokens=[EOT]
)


def text_iterator():
    for row in tqdm(fw, total=total_docs, unit="docs"):
        txt = row.get("text", "")
        if txt and isinstance(txt, str):
            yield txt


tokenizer.train_from_iterator(text_iterator(), trainer=trainer)


fast_tokenizer = PreTrainedTokenizerFast(
    tokenizer_object = tokenizer,
    eos_token = EOT,
    bos_token = None,
    unk_token = None
)

fast_tokenizer.save_pretrained("noomo")
