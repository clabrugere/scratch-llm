from typing import Callable

import pytest
import torch

from scratch_llm.model.cache import KVCache


@pytest.fixture
def make_cache() -> Callable[..., KVCache]:
    def inner(batch_size=1, num_layers=2, num_head=2, max_seq_len=64, head_dim=16) -> KVCache:
        return KVCache(num_layers, batch_size, num_head, max_seq_len, head_dim, torch.device("cpu"))

    return inner


@pytest.mark.parametrize("batch_size", [1, 4])
def test_kv_cache(make_cache, batch_size):
    num_layers = 2
    cache = make_cache(batch_size=batch_size)

    assert len(cache) == num_layers

    # prefill with 8 tokens
    for layer_cache in cache:
        k_new = torch.randn(batch_size, 2, 8, 16)
        v_new = torch.randn(batch_size, 2, 8, 16)
        k, v = layer_cache.update(k_new, v_new)
        assert k.size() == (batch_size, 2, 8, 16)
        assert v.size() == (batch_size, 2, 8, 16)
    cache.step(8)

    assert cache.current_seq_len == 8

    # decode step with 1 new token
    for layer_cache in cache:
        k_new = torch.randn(batch_size, 2, 1, 16)
        v_new = torch.randn(batch_size, 2, 1, 16)
        k, v = layer_cache.update(k_new, v_new)
        assert k.size() == (batch_size, 2, 9, 16)
        assert v.size() == (batch_size, 2, 9, 16)
    cache.step(1)

    assert cache.current_seq_len == 9


def test_kv_cache_value_correctness(make_cache):
    cache = make_cache()
    k_new = torch.randn(1, 2, 8, 16)
    v_new = torch.randn(1, 2, 8, 16)

    k, v = cache.update(0, k_new, v_new)

    torch.testing.assert_close(k, k_new)
    torch.testing.assert_close(v, v_new)


def test_kv_cache_decode_preserves_prefill_values(make_cache):
    cache = make_cache()
    k_prefill = torch.randn(1, 2, 8, 16)
    v_prefill = torch.randn(1, 2, 8, 16)
    cache.update(0, k_prefill, v_prefill)
    cache.step(8)

    k_decode = torch.randn(1, 2, 1, 16)
    v_decode = torch.randn(1, 2, 1, 16)
    k, v = cache.update(0, k_decode, v_decode)

    # prefill tokens are unchanged
    torch.testing.assert_close(k[:, :, :8, :], k_prefill)
    torch.testing.assert_close(v[:, :, :8, :], v_prefill)
    # new token is appended correctly
    torch.testing.assert_close(k[:, :, 8:, :], k_decode)
    torch.testing.assert_close(v[:, :, 8:, :], v_decode)


def test_kv_cache_update_without_step_overwrites(make_cache):
    # update() without step() overwrites: second write replaces first at position 0
    cache = make_cache()
    k_first = torch.ones(1, 2, 1, 16)
    k_second = torch.full((1, 2, 1, 16), 2.0)

    cache.update(0, k_first, torch.zeros_like(k_first))
    k, _ = cache.update(0, k_second, torch.zeros_like(k_second))

    torch.testing.assert_close(k[:, :, 0:1, :], k_second)


@pytest.mark.parametrize("num_steps", [1, 5])
def test_kv_cache_multiple_decode_steps(make_cache, num_steps):
    prefill_len = 4
    cache = make_cache(max_seq_len=prefill_len + num_steps)

    k_prefill = torch.randn(1, 2, prefill_len, 16)
    v_prefill = torch.randn(1, 2, prefill_len, 16)
    cache.update(0, k_prefill, v_prefill)
    cache.step(prefill_len)

    decode_keys = []
    for _ in range(num_steps):
        k_new = torch.randn(1, 2, 1, 16)
        v_new = torch.randn(1, 2, 1, 16)
        k, _ = cache.update(0, k_new, v_new)
        decode_keys.append(k_new)
        cache.step(1)

    assert k.size(2) == prefill_len + num_steps

    # prefill tokens preserved
    torch.testing.assert_close(k[:, :, :prefill_len, :], k_prefill)

    # all decode tokens preserved in order
    for i, dk in enumerate(decode_keys):
        torch.testing.assert_close(k[:, :, prefill_len + i : prefill_len + i + 1, :], dk)


def test_kv_cache_reset(make_cache):
    cache = make_cache()
    cache.update(0, torch.randn(1, 2, 8, 16), torch.randn(1, 2, 8, 16))
    cache.step(8)

    cache.reset()

    assert cache.current_seq_len == 0

    k_new = torch.randn(1, 2, 3, 16)
    v_new = torch.randn(1, 2, 3, 16)
    k, _ = cache.update(0, k_new, v_new)

    assert k.size(2) == 3
    torch.testing.assert_close(k, k_new)


def test_kv_cache_layer_isolation(make_cache):
    cache = make_cache(num_layers=2)
    k0 = torch.ones(1, 2, 4, 16)
    k1 = torch.full((1, 2, 4, 16), 2.0)

    cache.update(0, k0, torch.zeros_like(k0))
    k_layer1, _ = cache.update(1, k1, torch.zeros_like(k1))

    # layer 1 output reflects only what was written to layer 1
    torch.testing.assert_close(k_layer1, k1)

    # layer 0 output is unaffected
    k_layer0, _ = cache.update(0, k0, torch.zeros_like(k0))
    torch.testing.assert_close(k_layer0, k0)


def test_kv_cache_fill_to_capacity(make_cache):
    max_seq_len = 16
    cache = make_cache(max_seq_len=max_seq_len)

    k_full = torch.randn(1, 2, max_seq_len, 16)
    v_full = torch.randn(1, 2, max_seq_len, 16)
    k, _ = cache.update(0, k_full, v_full)

    assert k.size() == (1, 2, max_seq_len, 16)
    torch.testing.assert_close(k[:, :, -1:, :], k_full[:, :, -1:, :])


def test_layer_kv_cache_current_seq_len(make_cache):
    cache = make_cache()
    layer_cache = next(iter(cache))

    assert layer_cache.current_seq_len == 0

    cache.step(5)
    assert layer_cache.current_seq_len == 5
