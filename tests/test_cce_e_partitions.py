"""Striped dE buffers preserve row addressing and the probability filter."""

import pytest
import torch
import triton

from cut_cross_entropy.cce_backward import cce_backward_kernel
from cut_cross_entropy.cce_lse_forward import cce_lse_forward_kernel
from cut_cross_entropy.utils import TensorInfo, _handle_eps

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Test requires CUDA")


@pytest.mark.parametrize("partitions", [2, 4, 8])
@pytest.mark.parametrize("selected_rows", [False, True])
@pytest.mark.parametrize("fp32_partials", [False, True])
def test_e_partitions_addressing_filter_and_gradient(partitions, selected_rows, fp32_partials):
    torch.manual_seed(1067)
    b, v, d = 131, 2049, 64
    e = torch.randn(b, d, device="cuda", dtype=torch.bfloat16) * 0.125
    c = torch.randn(v, d, device="cuda", dtype=torch.bfloat16) * 0.125
    targets = torch.randint(v, (b,), device="cuda")
    order = torch.randperm(v, device="cuda", dtype=torch.int32)
    valids = torch.arange(0, b, 2, device="cuda") if selected_rows else None
    ret = cce_lse_forward_kernel(
        e,
        c,
        targets=targets,
        valids=valids,
        vocab_ordering=order,
        return_row_max=True,
    )
    rows = ret.lse.numel()
    # Forward LSE is compact; backward dLSE is addressed in the original row
    # space (the public wrapper expands LSE before exposing it to autograd).
    dlse = torch.zeros(b, device="cuda")
    if valids is None:
        dlse.copy_(2e-4 * ret.lse / rows)
    else:
        dlse[valids] = 2e-4 * ret.lse / rows
    kwargs = dict(
        do=torch.ones((), device="cuda"),
        dlse=dlse,
        e=e,
        e_info=TensorInfo(torch.float32, True),
        c=c,
        c_info=TensorInfo(torch.float32, True),
        bias=None,
        bias_info=None,
        lse=ret.lse,
        valids=valids,
        softcap=None,
        filter_eps=_handle_eps("auto", e.dtype),
        targets=targets,
        vocab_ordering=order,
        row_max=ret.row_max,
        neg_correct_logit=ret.neg_correct_logit,
        target_tile=ret.target_tile,
        grad_scale=1 / rows,
        accum_c_fp32=True,
    )
    flags_reference = torch.empty(
        (triton.cdiv(rows, 128), triton.cdiv(v, 128)),
        device="cuda",
        dtype=torch.int32,
    )
    flags = torch.empty_like(flags_reference)
    de_reference, dc_reference, _ = cce_backward_kernel(
        **kwargs,
        accum_e_fp32=True,
        tile_flags=flags_reference,
    )
    de, dc, _ = cce_backward_kernel(
        **kwargs,
        accum_e_fp32=fp32_partials,
        e_grad_partitions=partitions,
        tile_flags=flags,
    )
    torch.testing.assert_close(flags, flags_reference, rtol=0, atol=0)
    torch.testing.assert_close(dc, dc_reference, rtol=1e-5, atol=1e-6)
    assert torch.isfinite(de).all()
    if fp32_partials:
        torch.testing.assert_close(de, de_reference, rtol=1e-5, atol=1e-6)
    else:
        assert ((de - de_reference).norm() / de_reference.norm()).item() < 1e-2
    if selected_rows:
        torch.testing.assert_close(de[1::2], torch.zeros_like(de[1::2]), rtol=0, atol=0)
