import numpy as np
import pytest

from srvar.api import fit
from srvar.data.dataset import Dataset
from srvar.elb import ElbSpec
from srvar.spec import ModelSpec, PriorSpec, SamplerConfig
from srvar.sv import VolatilitySpec


@pytest.mark.parametrize(
    "model",
    [
        ModelSpec(p=1, elb=ElbSpec(bound=0, applies_to=["a"])),
        ModelSpec(p=1, volatility=VolatilitySpec()),
    ],
)
def test_blasso_rejected_for_elb_and_sv(model):
    ds = Dataset.from_arrays(
        values=np.random.default_rng(1).normal(size=(20, 2)), variables=["a", "b"]
    )
    with pytest.raises(ValueError, match="Bayesian LASSO inference is disabled"):
        fit(ds, model, PriorSpec.from_blasso(k=3, n=2), SamplerConfig(draws=20, burn_in=5))
