"""Bounded production-BF16 gate using the upstream dense-gradient assertions.

Three original shapes cross eight profiles covering z-loss, shifted/ignored
rows, every reduction, and representative bias/softcap paths. This avoids the
upstream Cartesian product across unrelated implementations for kernel gates.
"""

import pytest
import torch
from test_cce_loss_backward import test_loss_backward as reference_case

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Test requires CUDA")

PROFILES = [
    (True, False, False, "mean", False, None),
    (False, False, False, "mean", False, None),
    (True, True, True, "mean", False, None),
    (False, True, True, "none", False, None),
    (True, False, True, "sum", False, None),
    (True, True, False, "none", False, None),
    (False, False, True, "none", True, None),
    (True, True, True, "sum", True, 20.0),
]


@pytest.mark.parametrize("shape", [(256, 512, 512), (252, 507, 512), (252, 507, 497)])
@pytest.mark.parametrize("z_loss,shift,invalids,reduction,has_bias,softcap", PROFILES)
def test_training_gradient_contract(shape, z_loss, shift, invalids, reduction, has_bias, softcap):
    reference_case(
        impl="cce",
        dtype=torch.bfloat16,
        error_tol=1e-2,
        softcap=softcap,
        has_bias=has_bias,
        shift=shift,
        invalids=invalids,
        reduction=reduction,
        z_loss=z_loss,
        shape=shape,
    )
