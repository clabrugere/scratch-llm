import pytest

from model.tokenizer import BPETokenizer, PairIndex, TokenSequence, string_to_byte

# string_to_byte


def test_string_to_byte_ascii():
    assert string_to_byte("abc", "utf-8") == [97, 98, 99]


def test_string_to_byte_multibyte():
    assert string_to_byte("é", "utf-8") == list("é".encode("utf-8"))


def test_string_to_byte_cjk():
    assert string_to_byte("中", "utf-8") == list("中".encode("utf-8"))


def test_string_to_byte_empty():
    assert string_to_byte("", "utf-8") == []


def test_string_to_byte_values_in_range():
    result = string_to_byte("Hello, 世界! 🎉", "utf-8")
    assert all(0 <= b <= 255 for b in result)


# TokenSequence


def test_token_sequence_iter_fresh():
    seq = TokenSequence([10, 20, 30])
    assert list(seq) == [(0, 10), (1, 20), (2, 30)]


def test_token_sequence_len():
    seq = TokenSequence([1, 2, 3, 4])
    assert len(seq) == 4


def test_token_sequence_boundary_has_left_right():
    seq = TokenSequence([1, 2, 3])
    assert not seq.has_left(0)
    assert seq.has_right(0)
    assert seq.has_left(2)
    assert not seq.has_right(2)


def test_token_sequence_delete_middle():
    seq = TokenSequence([10, 20, 30, 40])
    seq.delete(1)
    assert list(seq) == [(0, 10), (2, 30), (3, 40)]


def test_token_sequence_delete_last():
    seq = TokenSequence([10, 20, 30])
    seq.delete(2)
    assert list(seq) == [(0, 10), (1, 20)]


def test_token_sequence_delete_updates_left_of_right_neighbor():
    seq = TokenSequence([1, 2, 3])
    seq.delete(1)
    assert seq.left[2] == 0


def test_token_sequence_delete_multiple():
    seq = TokenSequence([1, 2, 3, 4, 5])
    seq.delete(1)
    seq.delete(3)
    assert list(seq) == [(0, 1), (2, 3), (4, 5)]


def test_token_sequence_single_element():
    seq = TokenSequence([42])
    assert list(seq) == [(0, 42)]
    assert not seq.has_left(0)
    assert not seq.has_right(0)


# PairIndex


def test_pair_index_initial_falsy():
    assert not PairIndex()


def test_pair_index_add_increments_count():
    idx = PairIndex()
    idx.add((1, 2), 0)
    assert idx.counts[(1, 2)] == 1
    assert 0 in idx.positions[(1, 2)]


def test_pair_index_add_same_pair_twice():
    idx = PairIndex()
    idx.add((1, 2), 0)
    idx.add((1, 2), 5)
    assert idx.counts[(1, 2)] == 2
    assert idx.positions[(1, 2)] == {0, 5}


def test_pair_index_truthy_after_add():
    idx = PairIndex()
    idx.add((1, 2), 0)
    assert bool(idx)


def test_pair_index_most_frequent():
    idx = PairIndex()
    idx.add((1, 2), 0)
    idx.add((1, 2), 1)  # count=2
    idx.add((3, 4), 2)  # count=1
    pair, count = idx.most_frequent()
    assert pair == (1, 2)
    assert count == 2


def test_pair_index_most_frequent_skips_stale():
    idx = PairIndex()
    idx.add((1, 2), 0)
    idx.add((1, 2), 1)  # heap has stale (-2, (1,2)) after this remove:
    idx.remove((1, 2), 0)  # count drops to 1
    pair, count = idx.most_frequent()
    assert pair == (1, 2)
    assert count == 1


def test_pair_index_remove_decrements():
    idx = PairIndex()
    idx.add((1, 2), 0)
    idx.add((1, 2), 5)
    idx.remove((1, 2), 0)
    assert idx.counts[(1, 2)] == 1
    assert 0 not in idx.positions[(1, 2)]


def test_pair_index_remove_to_zero_deletes_pair():
    idx = PairIndex()
    idx.add((1, 2), 0)
    idx.remove((1, 2), 0)
    assert (1, 2) not in idx.counts
    assert not bool(idx)


def test_pair_index_pop_returns_positions_and_removes():
    idx = PairIndex()
    idx.add((1, 2), 0)
    idx.add((1, 2), 3)
    positions = idx.pop((1, 2))
    assert positions == {0, 3}
    assert (1, 2) not in idx.counts
    assert not bool(idx)


# BPETokenizer initialization


def test_tokenizer_max_vocab_256_raises():
    with pytest.raises(ValueError):
        BPETokenizer(256)


def test_tokenizer_max_vocab_zero_raises():
    with pytest.raises(ValueError):
        BPETokenizer(0)


def test_tokenizer_minimal_valid_size():
    t = BPETokenizer(257)
    assert t.vocab_size == 256


def test_tokenizer_initial_vocab_covers_all_bytes():
    t = BPETokenizer(300)
    for i in range(256):
        assert t.id_to_token[i] == bytes([i])


# BPETokenizer reset


def test_tokenizer_reset_after_train():
    t = BPETokenizer(260)
    t.train("aaaa")
    assert t.vocab_size > 256
    t.reset()
    assert t.vocab_size == 256
    assert t.pairs == {}


def test_tokenizer_reset_clears_special_tokens():
    t = BPETokenizer(300)
    t.register_special_token("<BOS>")
    t.reset()
    assert t.special_to_id == {}
    assert t.vocab_size == 256


# BPETokenizer register_special_token


def test_register_special_token_id_at_least_256():
    t = BPETokenizer(300)
    assert t.register_special_token("<BOS>") >= 256


def test_register_special_token_idempotent():
    t = BPETokenizer(300)
    assert t.register_special_token("<BOS>") == t.register_special_token("<BOS>")


def test_register_special_token_different_ids():
    t = BPETokenizer(300)
    assert t.register_special_token("<BOS>") != t.register_special_token("<EOS>")


def test_register_special_token_increments_vocab_size():
    t = BPETokenizer(300)
    t.register_special_token("<BOS>")
    assert t.vocab_size == 257
    t.register_special_token("<EOS>")
    assert t.vocab_size == 258


def test_register_special_token_idempotent_no_vocab_growth():
    t = BPETokenizer(300)
    t.register_special_token("<BOS>")
    t.register_special_token("<BOS>")
    assert t.vocab_size == 257


# BPETokenizer train


def test_train_empty_no_merges():
    t = BPETokenizer(300)
    t.train("")
    assert t.vocab_size == 256


def test_train_single_char_no_merges():
    t = BPETokenizer(300)
    t.train("a")
    assert t.vocab_size == 256


def test_train_stop_early_on_unique_pairs():
    # every pair in "abcde" appears once → stop_early halts before any merge
    t = BPETokenizer(300)
    t.train("abcde", stop_early=True)
    assert t.vocab_size == 256


def test_train_stop_early_fewer_merges():
    # without stop_early, unique-pair input is fully merged up to max_vocab_size
    t_early = BPETokenizer(300)
    t_full = BPETokenizer(300)
    t_early.train("abcde", stop_early=True)
    t_full.train("abcde", stop_early=False)
    assert t_early.vocab_size < t_full.vocab_size


def test_train_repeated_pair_is_merged():
    t = BPETokenizer(257)
    t.train("aaaa")
    assert (97, 97) in t.pairs
    assert t.vocab_size == 257


def test_train_merge_token_is_concatenation():
    t = BPETokenizer(257)
    t.train("aaaa")
    mid = t.pairs[(97, 97)]
    assert t.id_to_token[mid] == t.id_to_token[97] + t.id_to_token[97]


def test_train_respects_max_vocab_size():
    t = BPETokenizer(258)
    t.train("aaaa")
    # (97,97)→256 then (256,256)→257 — both merges fit exactly
    assert t.vocab_size == 258
    assert t.id_to_token[257] == b"aaaa"


def test_train_unicode():
    t = BPETokenizer(300)
    t.train("中中中中")
    assert t.vocab_size > 256


# BPETokenizer encode


def test_encode_untrained_ascii():
    t = BPETokenizer(300)
    assert t.encode("abc") == [97, 98, 99]


def test_encode_empty():
    t = BPETokenizer(300)
    assert t.encode("") == []


def test_encode_single_char():
    t = BPETokenizer(300)
    assert t.encode("a") == [97]


def test_encode_uses_merges():
    t = BPETokenizer(257)
    t.train("aaaa")
    assert t.encode("aa") == [256]


def test_encode_two_merges():
    t = BPETokenizer(258)
    t.train("aaaa")
    assert t.encode("aaaa") == [257]
    assert t.encode("aa") == [256]
    assert t.encode("a") == [97]


def test_encode_special_token_only():
    t = BPETokenizer(300)
    bos_id = t.register_special_token("<BOS>")
    assert t.encode("<BOS>") == [bos_id]


def test_encode_special_tokens_in_text():
    t = BPETokenizer(300)
    bos_id = t.register_special_token("<BOS>")
    eos_id = t.register_special_token("<EOS>")
    result = t.encode("<BOS>hi<EOS>")
    assert result[0] == bos_id
    assert result[-1] == eos_id
    assert result[1:-1] == [ord("h"), ord("i")]


def test_encode_longest_match_special_tokens():
    # "<BOS>" is a prefix of "<BOSE>"; longest-first must prevent shadowing
    t = BPETokenizer(300)
    bos_id = t.register_special_token("<BOS>")
    bose_id = t.register_special_token("<BOSE>")
    assert t.encode("<BOSE>") == [bose_id]
    assert t.encode("<BOS>") == [bos_id]


# BPETokenizer — decode


def test_decode_raw_bytes():
    t = BPETokenizer(300)
    assert t.decode([97, 98, 99]) == "abc"


def test_decode_empty():
    t = BPETokenizer(300)
    assert t.decode([]) == ""


def test_decode_special_token():
    t = BPETokenizer(300)
    bos_id = t.register_special_token("<BOS>")
    assert t.decode([bos_id]) == "<BOS>"


def test_decode_merged_token():
    t = BPETokenizer(257)
    t.train("aaaa")
    assert t.decode([256]) == "aa"


# decode(encode(text)) == text


def assert_roundtrip(tokenizer: BPETokenizer, text: str) -> None:
    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_roundtrip_ascii():
    assert_roundtrip(BPETokenizer(300), "Hello, world!")


def test_roundtrip_empty():
    assert_roundtrip(BPETokenizer(300), "")


def test_roundtrip_unicode():
    assert_roundtrip(BPETokenizer(300), "中文🎉")


def test_roundtrip_multibyte_latin():
    assert_roundtrip(BPETokenizer(300), "naïve café résumé")


def test_roundtrip_special_tokens():
    t = BPETokenizer(300)
    t.register_special_token("<BOS>")
    t.register_special_token("<EOS>")
    assert_roundtrip(t, "<BOS>Hello, world!<EOS>")


def test_roundtrip_after_training():
    t = BPETokenizer(300)
    corpus = "the quick brown fox jumps over the lazy dog " * 5
    t.train(corpus)
    assert_roundtrip(t, "the quick brown fox")


def test_roundtrip_unicode_after_training():
    t = BPETokenizer(300)
    t.train("中文中文中文中文中文中文")
    assert_roundtrip(t, "中文")


def test_roundtrip_trained_with_special_tokens():
    t = BPETokenizer(300)
    t.register_special_token("<BOS>")
    t.register_special_token("<EOS>")
    t.train("hello world " * 10)
    assert_roundtrip(t, "<BOS>hello world<EOS>")
