
import torch
from torch.utils.data import Dataset, DataLoader

from model_seq2seq import Seq2SeqTransformer, Seq2SeqConfig
from transformers import GPT2TokenizerFast


device = "cuda" if torch.cuda.is_available() else "cpu"


class PairSeq2SeqDataset(Dataset):

    def __init__(self, pairs, tokenizer, max_encoder_len=64, max_decoder_len=64):
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.max_encoder_len = max_encoder_len
        self.max_decoder_len = max_decoder_len

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


    pairs = [
        ("question: what is 2 plus 2", "4"),
        ("question: what color is the sky", "blue"),
        ("User: hello", "Assistant: hi"),
        ("User: how are you", "Assistant: i am fine"),
    ]

    tokenizer = GPT2TokenizerFast.from_pretrained("data/gpt2", local_files_only=True)

    train_dataset = PairSeq2SeqDataset(
        pairs=pairs,
        tokenizer=tokenizer,
        max_encoder_len=64,
        max_decoder_len=64,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=True,
        collate_fn=lambda batch: collate_seq2seq_batch(
            batch,
            pad_token_id=tokenizer.pad_token_id
        )
    )

    config = Seq2SeqConfig(
        block_size=128,
        vocab_size=tokenizer.vocab_size,
        n_layer=8,
        n_head=8,
        n_embd=128,
        flash_attn=True,
        use_rope=True,
    )

    model = Seq2SeqTransformer(config)

    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # train loop
    model.train()

    for epoch in range(20):
        total_loss = 0.0

        for batch in train_loader:
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

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"epoch={epoch+1} loss={avg_loss:.4f}")

    # validation
    model.eval()

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
        pad_token_id=tokenizer.pad_token_id,
        do_sample=False,
    )

    print(generated)
    print(tokenizer.decode(generated[0].tolist(), skip_special_tokens=True))
