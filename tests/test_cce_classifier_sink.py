"""Persistent classifier gradients must accumulate once across backwards."""

import pytest
import torch

from cut_cross_entropy.cce import CCEParams, linear_cross_entropy_apply
from cut_cross_entropy.cce_backward import cce_backward_kernel
from cut_cross_entropy.cce_lse_forward import cce_lse_forward_kernel
from cut_cross_entropy.utils import TensorInfo, _handle_eps, compute_z_loss

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Test requires CUDA")


@pytest.mark.parametrize("atomic", [False, True])
def test_classifier_sink_preserves_fp32_reference_and_existing_contents(atomic, monkeypatch):
    torch.manual_seed(1033)
    b, v, d = 257, 1025, 128
    e = torch.randn(b, d, device="cuda", dtype=torch.bfloat16) * 0.125
    c = torch.randn(v, d, device="cuda", dtype=torch.bfloat16) * 0.125
    targets = torch.randint(v, (b,), device="cuda")
    order = torch.randperm(v, device="cuda", dtype=torch.int32)
    ret = cce_lse_forward_kernel(
        e,
        c,
        targets=targets,
        vocab_ordering=order,
        return_row_max=True,
    )
    kwargs = dict(
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
        filter_eps=_handle_eps("auto", e.dtype),
        targets=targets,
        vocab_ordering=order,
        row_max=ret.row_max,
        neg_correct_logit=ret.neg_correct_logit,
        target_tile=ret.target_tile,
        grad_scale=1 / b,
        accum_e_fp32=True,
        accum_c_fp32=True,
    )
    de_reference, dc_reference, _ = cce_backward_kernel(**kwargs)
    sink = torch.zeros(c.shape, device="cuda", dtype=torch.float32)
    for count in (1, 2):
        de, dc, _ = cce_backward_kernel(
            **kwargs, classifier_grad_sink=sink, classifier_sink_atomic=atomic
        )
        assert dc is None
        torch.testing.assert_close(de, de_reference, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(sink, count * dc_reference, rtol=1e-5, atol=1e-6)
    with pytest.raises(ValueError, match="FP32"):
        cce_backward_kernel(**kwargs, classifier_grad_sink=sink.to(torch.bfloat16))
    monkeypatch.setattr("cut_cross_entropy.cce_backward.cce_fixed_block_shape", lambda _: None)
    with pytest.raises(ValueError, match="CCE_AUTOTUNE"):
        cce_backward_kernel(**kwargs, classifier_grad_sink=sink)


@pytest.mark.parametrize("atomic", [False, True])
def test_classifier_sink_autograd_does_not_return_a_second_classifier_gradient(atomic):
    torch.manual_seed(1041)
    e = torch.randn(129, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    c = torch.randn(257, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    targets = torch.randint(c.shape[0], (e.shape[0],), device="cuda")
    sink = torch.zeros(c.shape, device="cuda", dtype=torch.float32)
    params = CCEParams(
        targets=targets,
        valids=None,
        softcap=None,
        reduction="mean",
        filter_eps=_handle_eps("auto", e.dtype),
        shift=0,
        batch_shape=targets.shape,
        accum_e_fp32=False,
        accum_c_fp32=False,
        filter_e_grad=True,
        filter_c_grad=True,
        vocab_parallel_options=None,
        return_lse=True,
        vocab_ordering=torch.arange(c.shape[0], device="cuda", dtype=torch.int32),
        classifier_grad_sink=sink,
        classifier_sink_atomic=atomic,
    )
    first = None
    for count in (1, 2):
        ce, lse = linear_cross_entropy_apply(e, c, None, params)
        (ce + 1e-4 * compute_z_loss(lse)).backward()
        assert c.grad is None
        assert e.grad is not None and torch.isfinite(e.grad).all()
        if first is None:
            first = sink.clone()
            assert first.norm() > 0
        else:
            torch.testing.assert_close(sink, count * first, rtol=1e-5, atol=1e-6)
