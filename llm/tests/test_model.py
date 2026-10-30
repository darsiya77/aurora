import torch

from aurora_llm.model import GPT


def test_shapes_and_loss(small_cfg):
    m = GPT(small_cfg)
    x = torch.randint(0, small_cfg.vocab_size, (3, 16))
    y = torch.randint(0, small_cfg.vocab_size, (3, 16))
    logits, loss = m(x, y)
    assert logits.shape == (3, 16, small_cfg.vocab_size)
    # An untrained model is close to a uniform guess on random targets: loss ~ ln(vocab).
    assert abs(loss.item() - torch.log(torch.tensor(97.0)).item()) < 0.5


def test_causal_future_tokens_do_not_leak(small_cfg):
    m = GPT(small_cfg).eval()
    x = torch.randint(0, small_cfg.vocab_size, (1, 12))
    y = x.clone()
    y[0, 8:] = (y[0, 8:] + 1) % small_cfg.vocab_size
    a, _ = m(x, x)
    b, _ = m(y, y)
    assert torch.allclose(a[0, :8], b[0, :8], atol=1e-5)


def test_kv_cache_matches_full_forward(small_cfg):
    m = GPT(small_cfg).eval()
    x = torch.randint(0, small_cfg.vocab_size, (1, 10))
    full, _ = m(x, x)
    caches = [dict() for _ in m.blocks]
    m(x[:, :6], caches=caches, start_pos=0)
    for i in range(6, 10):
        step, _ = m(x[:, i:i + 1], caches=caches, start_pos=i)
        assert torch.allclose(step[0, -1], full[0, i], atol=1e-4)


def test_cache_prefill_with_several_new_tokens(small_cfg):
    m = GPT(small_cfg).eval()
    x = torch.randint(0, small_cfg.vocab_size, (1, 10))
    full, _ = m(x, x)
    caches = [dict() for _ in m.blocks]
    m(x[:, :4], caches=caches, start_pos=0)
    out, _ = m(x[:, 4:10], caches=caches, start_pos=4)
    assert torch.allclose(out[0, -1], full[0, 9], atol=1e-4)


def test_loss_mask_ignores_masked_positions(small_cfg):
    m = GPT(small_cfg)
    x = torch.randint(0, small_cfg.vocab_size, (2, 8))
    mask = torch.zeros(2, 8, dtype=torch.long)
    mask[:, -2:] = 1
    _, l1 = m(x, x, loss_mask=mask)
    y = x.clone()
    y[:, :6] = 5  # targets where the mask is 0 must not affect the loss
    _, l2 = m(x, y, loss_mask=mask)
    assert torch.allclose(l1, l2)


def test_overfits_one_batch(small_cfg):
    m = GPT(small_cfg)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    x = torch.randint(0, small_cfg.vocab_size, (4, 16))
    for _ in range(150):
        _, loss = m(x[:, :-1], x[:, 1:])
        loss.backward()
        opt.step()
        opt.zero_grad()
    assert loss.item() < 0.3


def test_generate_respects_length_and_stop(small_cfg):
    m = GPT(small_cfg)
    x = torch.randint(0, small_cfg.vocab_size, (1, 5))
    out = m.generate(x, max_new_tokens=50, temperature=0)
    assert out.shape[1] <= small_cfg.max_seq_len
    first = out[0, 5].item()
    out2 = m.generate(x, max_new_tokens=10, temperature=0, stop_ids=(first,))
    assert out2.shape[1] == 6
