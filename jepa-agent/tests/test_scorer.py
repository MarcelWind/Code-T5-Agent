"""Test JEPA scorer."""

import sys
sys.path.insert(0, ".")

import numpy as np
from scorer import jepa_loss, cosine_distance, l2_distance, rank_candidates


def test_cosine_distance():
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([0.0, 1.0, 0.0])
    d = cosine_distance(a, b)
    assert abs(d - 1.0) < 0.01, f"Orthogonal vectors → distance 1, got {d}"
    print(f"  ✓ cosine_distance orthogonal = {d:.4f}")


def test_cosine_identical():
    a = np.array([1.0, 2.0, 3.0])
    d = cosine_distance(a, a)
    assert abs(d) < 0.001, f"Identical vectors → distance 0, got {d}"
    print(f"  ✓ cosine_distance identical = {d:.6f}")


def test_cosine_opposite():
    a = np.array([1.0, 0.0])
    b = np.array([-1.0, 0.0])
    d = cosine_distance(a, b)
    assert abs(d - 2.0) < 0.01, f"Opposite vectors → distance 2, got {d}"
    print(f"  ✓ cosine_distance opposite = {d:.4f}")


def test_l2_distance():
    a = np.array([0.0, 0.0])
    b = np.array([3.0, 4.0])
    d = l2_distance(a, b)
    assert abs(d - 5.0) < 0.01, f"3-4-5 triangle → L2=5, got {d}"
    print(f"  ✓ l2_distance = {d:.4f}")


def test_jepa_loss_cosine():
    a = np.random.randn(768)
    b = np.random.randn(768)
    loss = jepa_loss(a, b, "cosine")
    assert 0.0 <= loss <= 2.0, f"Cosine loss should be in [0,2], got {loss}"
    print(f"  ✓ jepa_loss(cosine) = {loss:.4f}")


def test_jepa_loss_l2():
    a = np.random.randn(768)
    b = np.random.randn(768)
    loss = jepa_loss(a, b, "l2")
    assert loss >= 0.0, f"L2 loss should be ≥0, got {loss}"
    print(f"  ✓ jepa_loss(l2) = {loss:.4f}")


def test_rank_candidates():
    # Create embeddings: candidate 0 is the best match
    target = np.array([1.0, 0.0, 0.0])
    predicted = [
        np.array([1.0, 0.1, 0.0]),   # close → should rank 0
        np.array([0.0, 1.0, 0.0]),   # far  → should rank 1
        np.array([-1.0, 0.0, 0.0]),  # far  → should rank 2
    ]
    actual = [target, target, target]
    ranked = rank_candidates(predicted, actual, "cosine")
    assert ranked[0] == 0, f"Best candidate should be idx 0, got {ranked}"
    print(f"  ✓ rank_candidates returns {ranked}")


if __name__ == "__main__":
    test_cosine_distance()
    test_cosine_identical()
    test_cosine_opposite()
    test_l2_distance()
    test_jepa_loss_cosine()
    test_jepa_loss_l2()
    test_rank_candidates()
    print("\nAll scorer tests passed ✓")
