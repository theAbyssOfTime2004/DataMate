from datamate.etl.build import _walk_in_norm_set


def test_walk_in_set_adds_initials_of_multi_word_labels() -> None:
    found = _walk_in_norm_set(["Khách lẻ", "lẻ", "Đại lý vãng lai"])
    assert {"khách lẻ", "lẻ", "kl", "dlvl"} <= found
    # Single-word labels add no one-letter initial.
    assert "l" not in found


def test_walk_in_set_empty() -> None:
    assert _walk_in_norm_set([]) == set()
