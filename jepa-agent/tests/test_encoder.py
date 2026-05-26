"""Test CodeT5+ encoder."""

import sys
sys.path.insert(0, ".")

from encoder import CodeEncoder


def test_encode_shape():
    enc = CodeEncoder()
    code = "def foo(x): return x + 1"
    emb = enc.encode(code)
    assert emb.ndim == 1, f"Expected 1-D, got {emb.ndim}-D"
    assert emb.shape[0] == 768, f"Expected dim 768, got {emb.shape[0]}"
    print(f"  ✓ encode() returns shape ({emb.shape[0]},)")


def test_encode_diff():
    enc = CodeEncoder()
    orig = "def add(a, b): return a - b"
    fixed = "def add(a, b): return a + b"
    delta = enc.encode_diff(orig, fixed)
    assert delta.ndim == 1
    assert delta.shape[0] == 768
    # delta should be non-zero (different code)
    assert np.linalg.norm(delta) > 0, "Expected non-zero diff embedding"
    print(f"  ✓ encode_diff() returns non-zero delta (norm={np.linalg.norm(delta):.4f})")


def test_similar_code_close():
    enc = CodeEncoder()
    a = "def hello(): print('hi')"
    b = "def hello(): print('hi')"
    emb_a = enc.encode(a)
    emb_b = enc.encode(b)
    from scorer import cosine_distance
    d = cosine_distance(emb_a, emb_b)
    assert d < 0.01, f"Identical code should have near-zero distance, got {d:.4f}"
    print(f"  ✓ identical code → cosine distance = {d:.6f}")


def test_different_code_far():
    enc = CodeEncoder()
    a = "def hello(): print('hi')"
    b = "class Calculator: pass"
    emb_a = enc.encode(a)
    emb_b = enc.encode(b)
    from scorer import cosine_distance
    d = cosine_distance(emb_a, emb_b)
    assert d > 0.01, f"Different code should have non-zero distance, got {d:.4f}"
    print(f"  ✓ different code → cosine distance = {d:.6f}")


if __name__ == "__main__":
    import numpy as np
    test_encode_shape()
    test_encode_diff()
    test_similar_code_close()
    test_different_code_far()
    print("\nAll encoder tests passed ✓")
