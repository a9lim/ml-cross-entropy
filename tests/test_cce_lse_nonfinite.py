"""The locked LSE reduction must not conceal nonfinite classifier activations."""

import pytest
import torch

from cut_cross_entropy.cce_backward import cce_backward_kernel
from cut_cross_entropy.cce_lse_forward import cce_lse_forward_kernel
from cut_cross_entropy.utils import TensorInfo

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Test requires CUDA")


@pytest.mark.parametrize("nonfinite", ["nan", "inf"])
def test_lse_preserves_nonfinite_rows_and_gradients(nonfinite):
    torch.manual_seed(945)
    b, v, d = 259, 4099, 128
    e = torch.randn(b, d, device="cuda", dtype=torch.bfloat16) * 0.125
    c = torch.randn(v, d, device="cuda", dtype=torch.bfloat16) * 0.125
    c[17, 0] = float(nonfinite)
    order = torch.randperm(v, device="cuda", dtype=torch.int32)
    targets = torch.randint(v, (b,), device="cuda")
    ret = cce_lse_forward_kernel(
        e, c, targets=targets, vocab_ordering=order, return_row_max=True
    )
    # Min/max logaddexp previously discarded NaN partials. A +inf logit may
    # produce either +inf or NaN through softmax arithmetic, but must be visible.
    dense_lse = torch.logsumexp((e @ c.T).float(), dim=1)
    torch.testing.assert_close(torch.isfinite(ret.lse), torch.isfinite(dense_lse))
    de, dc, _ = cce_backward_kernel(
        do=torch.ones((), device="cuda"),
        dlse=2e-4 * ret.lse / b,
        e=e,
        e_info=TensorInfo(torch.float32, True),
        c=c,
        c_info=TensorInfo(torch.float32, True),
        bias=None,
        bias_info=None,
        lse=ret.lse,
        valids=None,
        softcap=None,
        filter_eps=None,
        targets=targets,
        vocab_ordering=order,
        grad_scale=1 / b,
        accum_e_fp32=True,
        accum_c_fp32=True,
        filter_e_grad=False,
        filter_c_grad=False,
    )
    assert not torch.isfinite(de).all()
    assert not torch.isfinite(dc).all()
