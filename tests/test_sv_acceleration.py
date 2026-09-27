import numpy as np
import pytest

from srvar import sv


def test_numba_and_numpy_mixture_sampling_match(monkeypatch):
    pytest.importorskip("numba")
    assert sv._HAVE_NUMBA
    data_rng = np.random.default_rng(529)
    y = np.concatenate([data_rng.normal(size=1000), [-1000.0, 1000.0]])
    h = data_rng.normal(size=y.size)
    reference_rng = np.random.default_rng(951)
    accelerated_rng = np.random.default_rng(951)
    monkeypatch.setenv("SRVAR_USE_NUMBA", "0")
    expected = sv.sample_mixture_indicators(y_star=y, h=h, rng=reference_rng)
    monkeypatch.setenv("SRVAR_USE_NUMBA", "1")
    actual = sv.sample_mixture_indicators(y_star=y, h=h, rng=accelerated_rng)
    np.testing.assert_array_equal(actual, expected)
    assert reference_rng.random() == accelerated_rng.random()
