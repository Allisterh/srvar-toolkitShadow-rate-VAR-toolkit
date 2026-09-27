import numpy as np
import pytest

from srvar.api import fit
from srvar.data.dataset import Dataset
from srvar.spec import ModelSpec, PriorSpec, SamplerConfig


@pytest.mark.parametrize("mode", ["global", "adaptive"])
def test_blasso_fit_rejected_before_sampling(mode):
    ds = Dataset.from_arrays(
        values=np.random.default_rng(1).normal(size=(20, 2)), variables=["a", "b"]
    )
    rng = np.random.default_rng(999)
    before = rng.bit_generator.state
    with pytest.raises(ValueError, match="Bayesian LASSO inference is disabled"):
        fit(
            ds,
            ModelSpec(p=1),
            PriorSpec.from_blasso(k=3, n=2, mode=mode),
            SamplerConfig(draws=20, burn_in=5),
            rng=rng,
        )
    assert rng.bit_generator.state == before
