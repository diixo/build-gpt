from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from model_seq2seq import Seq2SeqTransformer, Seq2SeqConfig
from transformers import GPT2TokenizerFast
from utils import plot_loss
from transformers import BartConfig, BartForConditionalGeneration


device = "cuda" if torch.cuda.is_available() else "cpu"


def load_text_pairs(path: str) -> List[Dict[str, str]]:

    items: List[Tuple[str, str]] = []

    user_text = None

    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()

            if not line:
                continue

            if line.startswith("User:"):
                user_text = line[len("User:"):].strip()

            elif line.startswith("Assistant:"):
                assistant_text = line[len("Assistant:"):].strip()

                if user_text is not None:
                    items.append(tuple([user_text, assistant_text]))

                user_text = None
            else:
                # any single line = separated knowledge-item
                items.append({
                    "knowledge": line,
                })

                # release current turn
                user_text = None
    return items


class PairSeq2SeqDataset(Dataset):

    def __init__(self,
        files: Union[str, Path, List[Union[str, Path]]],
        tokenizer,
        max_encoder_len=64,
        max_decoder_len=64
    ):
        self.tokenizer = tokenizer
        self.max_encoder_len = max_encoder_len
        self.max_decoder_len = max_decoder_len

        if isinstance(files, (str, Path)):
            files = [files]
        self.files = [Path(x) for x in files]

        self.pairs: List[Tuple[str, str]] = []
        for file_path in self.files:
            self.pairs.extend(load_text_pairs(str(file_path)))


    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        src_text, tgt_text = self.pairs[idx]

        # encoder side
        encoder_ids = self.tokenizer.encode(src_text)
        encoder_ids = encoder_ids[:self.max_encoder_len]

        # target side
        target_ids = self.tokenizer.encode(tgt_text)
        target_ids = target_ids[: self.max_decoder_len - 1]

        # append EOS to target
        target_ids = target_ids + [self.tokenizer.eos_token_id]

        # decoder input starts with BOS
        decoder_input_ids = [self.tokenizer.bos_token_id] + target_ids[:-1]
        targets = target_ids

        return {
            "encoder_input_ids": torch.tensor(encoder_ids, dtype=torch.long),
            "decoder_input_ids": torch.tensor(decoder_input_ids, dtype=torch.long),
            "targets": torch.tensor(targets, dtype=torch.long),
        }


def collate_seq2seq_batch(batch, pad_token_id: int):
    B = len(batch)

    max_src = max(x["encoder_input_ids"].size(0) for x in batch)
    max_dec = max(x["decoder_input_ids"].size(0) for x in batch)

    encoder_input_ids = torch.full((B, max_src), pad_token_id, dtype=torch.long)
    encoder_attention_mask = torch.zeros((B, max_src), dtype=torch.long)

    decoder_input_ids = torch.full((B, max_dec), pad_token_id, dtype=torch.long)
    decoder_attention_mask = torch.zeros((B, max_dec), dtype=torch.long)

    targets = torch.full((B, max_dec), -100, dtype=torch.long)

    for i, item in enumerate(batch):
        src = item["encoder_input_ids"]
        dec = item["decoder_input_ids"]
        tgt = item["targets"]

        encoder_input_ids[i, :src.size(0)] = src
        encoder_attention_mask[i, :src.size(0)] = 1

        decoder_input_ids[i, :dec.size(0)] = dec
        decoder_attention_mask[i, :dec.size(0)] = 1

        targets[i, :tgt.size(0)] = tgt

    return {
        "encoder_input_ids": encoder_input_ids,
        "encoder_attention_mask": encoder_attention_mask,
        "decoder_input_ids": decoder_input_ids,
        "decoder_attention_mask": decoder_attention_mask,
        "targets": targets,
    }



if __name__ == "__main__":

    EPOCHS = 20
    learning_rate = 5e-5
    BATCH_SIZE = 16

    tokenizer = GPT2TokenizerFast.from_pretrained("data/gpt-noomo-32k", local_files_only=True)

    config = Seq2SeqConfig(
        block_size=100,
        vocab_size=len(tokenizer.get_vocab()),  # size include special tokens
        n_layer=12,
        n_head=12,
        n_embd=576,
        flash_attn=True,
        use_rope=True,
    )

    train_dataset = PairSeq2SeqDataset(
        files = [
            "data/seq2seq_general_test_pairs_2k.txt",
        ],
        tokenizer = tokenizer,
        max_encoder_len = config.block_size,
        max_decoder_len = config.block_size,
    )

    print(f"train_dataset.size: {len(train_dataset)}")

    pad_token_id = tokenizer.eos_token_id if tokenizer.pad_token_id is None else tokenizer.pad_token_id

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=lambda batch: collate_seq2seq_batch(
            batch,
            pad_token_id=pad_token_id
        )
    )

    model = Seq2SeqTransformer(config)

    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    ### train loop
    losses = []

    model.train()

    for epoch in range(EPOCHS):
        epoch_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=True)

        for step, batch in enumerate(pbar, start=1):
            encoder_input_ids = batch["encoder_input_ids"].to(device)
            encoder_attention_mask = batch["encoder_attention_mask"].to(device)

            decoder_input_ids = batch["decoder_input_ids"].to(device)
            decoder_attention_mask = batch["decoder_attention_mask"].to(device)

            targets = batch["targets"].to(device)

            optimizer.zero_grad()

            out = model(
                encoder_input_ids=encoder_input_ids,
                decoder_input_ids=decoder_input_ids,
                encoder_attention_mask=encoder_attention_mask,
                decoder_attention_mask=decoder_attention_mask,
                targets=targets,
            )

            loss = out.loss
            loss.backward()
            optimizer.step()

            batch_item = loss.item()
            epoch_loss += batch_item
            running_avg_loss = epoch_loss / step

            pbar.set_postfix(
                batch_loss=f"{batch_item:.4f}",
                avg_loss=f"{running_avg_loss:.4f}"
            )

        avg_loss = epoch_loss / len(train_loader)
        losses.append(avg_loss)

        print(f"...epoch={epoch+1}, avg_loss={avg_loss:.4f}")

    ### validation
    model.eval()

    ############################################################################
    src_text = "User: hello"
    src_ids = tokenizer.encode(src_text)

    encoder_input_ids = torch.tensor([src_ids], dtype=torch.long, device=device)
    encoder_attention_mask = torch.ones_like(encoder_input_ids, device=device)

    generated = model.generate(
        encoder_input_ids=encoder_input_ids,
        encoder_attention_mask=encoder_attention_mask,
        max_new_tokens=10,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=pad_token_id,
        do_sample=False,
    )

    print(generated)
    print("Output text:", tokenizer.decode(generated[0].tolist(), skip_special_tokens=True))
    ############################################################################

    plot_loss(losses, type(model))
