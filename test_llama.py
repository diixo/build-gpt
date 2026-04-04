import torch
import pytest

from model_llama import GPTConfig, CausalSelfAttention, GPTLlama



def make_attn(flash_attn=False, n_head=2, n_embd=8, block_size=8):
    config = GPTConfig(
        block_size=block_size,
        vocab_size=100,
        n_layer=1,
        n_head=n_head,
        n_embd=n_embd,
        flash_attn=flash_attn,
        use_rope=False,
    )
    attn = CausalSelfAttention(config)
    attn.eval()
    return attn


@torch.no_grad()
def test_padding_does_not_change_real_token_outputs_manual_attention():
    """
    Проверяет главный инвариант:
    если добавить pad справа, выходы на реальных токенах не должны меняться.
    Этот тест как раз ловит ошибку полярности mask.
    """
    torch.manual_seed(42)

    attn = make_attn(flash_attn=False, block_size=8)

    B, T_real, T_pad, C = 1, 3, 5, attn.n_embd

    # один и тот же реальный контент
    x_real = torch.randn(B, T_real, C)

    # версия без паддинга
    mask_real = torch.tensor([[1, 1, 1]], dtype=torch.long)

    # версия с паддингом справа
    x_padded = torch.randn(B, T_pad, C)
    x_padded[:, :T_real, :] = x_real
    # pad-токены специально делаем большими/шумными, чтобы ошибка проявлялась сильнее
    x_padded[:, T_real:, :] = torch.randn(B, T_pad - T_real, C) * 100.0

    mask_padded = torch.tensor([[1, 1, 1, 0, 0]], dtype=torch.long)

    y_real = attn(x_real, attention_mask=mask_real)
    y_padded = attn(x_padded, attention_mask=mask_padded)

    assert torch.allclose(
        y_real,
        y_padded[:, :T_real, :],
        atol=1e-5,
        rtol=1e-5,
    ), "Padding влияет на выходы реальных токенов — маска, вероятно, сломана."


################ test-2
@torch.no_grad()
def test_real_positions_ignore_padded_keys():
    torch.manual_seed(123)

    attn = make_attn(flash_attn=False, block_size=8)
    B, T, C = 1, 5, attn.n_embd

    x1 = torch.randn(B, T, C)
    x2 = x1.clone()

    # меняем только pad-часть очень сильно
    x2[:, 3:, :] = torch.randn(B, 2, C) * 1000.0

    mask = torch.tensor([[1, 1, 1, 0, 0]], dtype=torch.long)

    y1 = attn(x1, attention_mask=mask)
    y2 = attn(x2, attention_mask=mask)

    # первые 3 позиции должны совпасть
    assert torch.allclose(
        y1[:, :3, :],
        y2[:, :3, :],
        atol=1e-5,
        rtol=1e-5,
    ), "Реальные токены зависят от pad-ключей — padding mask работает неверно."

### test-3
def build_expected_full_mask(attention_mask, T, device):
    # attention_mask: (B, T), где 1=real, 0=pad
    B = attention_mask.size(0)

    causal_mask = torch.triu(
        torch.ones(T, T, dtype=torch.bool, device=device),
        diagonal=1
    )[None, None, :, :]  # (1,1,T,T)

    key_padding_mask = (attention_mask == 0)[:, None, None, :]  # (B,1,1,T)

    return causal_mask | key_padding_mask


def test_full_mask_semantics():
    attention_mask = torch.tensor([
        [1, 1, 1, 0, 0],
        [1, 1, 0, 0, 0],
    ], dtype=torch.long)

    full_mask = build_expected_full_mask(attention_mask, T=5, device=attention_mask.device)

    # Проверим несколько точек руками:
    # batch 0, query position 0:
    # может смотреть только в key 0, key 1/2 запрещены causal, key 3/4 запрещены pad
    assert full_mask[0, 0, 0, 0].item() is False
    assert full_mask[0, 0, 0, 1].item() is True
    assert full_mask[0, 0, 0, 3].item() is True
    assert full_mask[0, 0, 0, 4].item() is True

    # batch 0, query position 2:
    # key 0,1,2 разрешены; key 3,4 запрещены как pad
    assert full_mask[0, 0, 2, 0].item() is False
    assert full_mask[0, 0, 2, 1].item() is False
    assert full_mask[0, 0, 2, 2].item() is False
    assert full_mask[0, 0, 2, 3].item() is True
    assert full_mask[0, 0, 2, 4].item() is True

### test-4
@torch.no_grad()
@pytest.mark.parametrize("flash_attn", [False, True])
def test_padding_does_not_change_real_outputs_both_paths(flash_attn):
    torch.manual_seed(7)

    attn = make_attn(flash_attn=flash_attn, block_size=8)
    B, T_real, T_pad, C = 1, 3, 5, attn.n_embd

    x_real = torch.randn(B, T_real, C)

    x_padded = torch.randn(B, T_pad, C)
    x_padded[:, :T_real, :] = x_real
    x_padded[:, T_real:, :] = torch.randn(B, T_pad - T_real, C) * 100.0

    mask_real = torch.tensor([[1, 1, 1]], dtype=torch.long)
    mask_padded = torch.tensor([[1, 1, 1, 0, 0]], dtype=torch.long)

    y_real = attn(x_real, attention_mask=mask_real)
    y_padded = attn(x_padded, attention_mask=mask_padded)

    assert torch.allclose(
        y_real,
        y_padded[:, :T_real, :],
        atol=1e-5,
        rtol=1e-5,
    )


### test-5 (global test)
@torch.no_grad()
def test_model_logits_invariant_to_right_padding():
    torch.manual_seed(1234)

    config = GPTConfig(
        block_size=8,
        vocab_size=50,
        n_layer=2,
        n_head=2,
        n_embd=8,
        flash_attn=False,
        use_rope=False,
    )
    model = GPTLlama(config)
    model.eval()

    idx_real = torch.tensor([[5, 9, 13]], dtype=torch.long)
    idx_pad  = torch.tensor([[5, 9, 13, 0, 0]], dtype=torch.long)

    mask_real = torch.tensor([[1, 1, 1]], dtype=torch.long)
    mask_pad  = torch.tensor([[1, 1, 1, 0, 0]], dtype=torch.long)

    out_real = model(idx_real, attention_mask=mask_real).logits
    out_pad  = model(idx_pad, attention_mask=mask_pad).logits

    assert torch.allclose(
        out_real,
        out_pad[:, :3, :],
        atol=1e-5,
        rtol=1e-5,
    ), "Logits на реальных токенах изменились из-за pad-хвоста."


if __name__ == "__main__":

    # test-1
    test_padding_does_not_change_real_token_outputs_manual_attention()

    # test-2
    test_real_positions_ignore_padded_keys()

    # test-3
    test_full_mask_semantics()

    # test-4
    #test_padding_does_not_change_real_outputs_both_paths(flash_attn=True)

    test_padding_does_not_change_real_outputs_both_paths(flash_attn=False)

    # test-5
    test_model_logits_invariant_to_right_padding()
