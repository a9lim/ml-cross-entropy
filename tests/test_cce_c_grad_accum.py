"""A caller-owned classifier gradient buffer accumulates across calls."""

import pytest
import torch

from cut_cross_entropy.cce import CCEParams, linear_cross_entropy_apply
from cut_cross_entropy.cce_backward import cce_backward_kernel
from cut_cross_entropy.cce_lse_forward import cce_lse_forward_kernel
from cut_cross_entropy.utils import TensorInfo, _handle_eps

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Test requires CUDA")


def _params(targets, ordering=None, accum=None):
    return CCEParams(
        targets=targets,
        valids=None,
        softcap=None,
        reduction="mean",
        filter_eps=_handle_eps("auto", torch.bfloat16),
        shift=0,
        batch_shape=targets.shape,
        accum_e_fp32=False,
        accum_c_fp32=False,
        filter_e_grad=True,
        filter_c_grad=True,
        vocab_parallel_options=None,
        return_lse=True,
        vocab_ordering=ordering,
        c_grad_accum=accum,
    )


def _operands(rows, vocab, dim, seed):
    torch.manual_seed(seed)
    e = (torch.randn(rows, dim, device="cuda") * 0.5).to(torch.bfloat16)
    c = (torch.randn(vocab, dim, device="cuda") * 0.5).to(torch.bfloat16)
    targets = torch.randint(vocab, (rows,), device="cuda")
    return e, c, targets


def _ordering(e, c):
    mean = e.float().mean(0, keepdim=True).to(c.dtype)
    return torch.argsort(mean @ c.mT, stable=True)[0].to(torch.int32)


def _run(e, c, targets, ordering, accum):
    embeddings = e.clone().requires_grad_()
    classifier = c.clone()
    if accum is None:
        classifier.requires_grad_()
    loss, _ = linear_cross_entropy_apply(
        embeddings, classifier, None, _params(targets, ordering, accum)
    )
    loss.backward()
    return loss.detach(), embeddings.grad, classifier.grad


def test_buffer_receives_the_kernels_gradient_exactly():
    """One forward, two backwards over one token tile: the destination is the
    only difference, so a zeroed buffer must come out bit-identical to the
    tensor the default path allocates, and a second backward must add to it
    rather than replace it. (Two separate forwards could not be compared this
    way: the LSE lock combines its vocabulary tiles in an arbitrary order.)"""
    rows, vocab, dim = 128, 512, 64
    e, c, targets = _operands(rows, vocab, dim, 7)
    order = torch.randperm(vocab, device="cuda").to(torch.int32)
    forward = cce_lse_forward_kernel(
        e,
        c,
        targets=targets,
        valids=None,
        shift=0,
        vocab_ordering=order,
        return_row_max=True,
    )
    arguments = dict(
        do=torch.ones((), device="cuda"),
        dlse=None,
        e=e,
        e_info=TensorInfo(torch.bfloat16, True),
        c=c,
        c_info=TensorInfo(torch.bfloat16, True),
        bias=None,
        bias_info=None,
        lse=forward.lse,
        valids=None,
        softcap=None,
        filter_eps=1e-4,
        targets=targets,
        shift=0,
        vocab_ordering=order,
        row_max=forward.row_max,
        neg_correct_logit=forward.neg_correct_logit,
        target_tile=forward.target_tile,
        grad_scale=1 / rows,
        accum_e_fp32=False,
        accum_c_fp32=False,
    )
    _, expected, _ = cce_backward_kernel(**arguments)
    assert expected.abs().sum() > 0

    accum = torch.zeros_like(c)
    _, returned, _ = cce_backward_kernel(**arguments, c_grad_accum=accum)
    assert returned is None
    torch.testing.assert_close(accum, expected, rtol=0, atol=0)

    cce_backward_kernel(**arguments, c_grad_accum=accum)
    torch.testing.assert_close(accum, expected + expected, rtol=0, atol=0)


def test_buffer_accumulates_across_calls():
    """Four consecutive calls into one buffer carry all four gradients.

    The reference is the FP32 sum the old caller kept, so the tolerance is the
    BF16 rounding the buffer introduces -- the point here is that nothing is
    dropped or overwritten, not how finely it is summed."""
    vocab, dim = 512, 64
    running = None
    accum = None
    for index in range(4):
        e, c, targets = _operands(128, vocab, dim, 11 + index)
        if accum is None:
            accum = torch.zeros_like(c)
        ordering = _ordering(e, c)
        _, _, dc = _run(e, c, targets, ordering, None)
        contribution = dc.float()
        running = contribution if running is None else running + contribution
        _run(e, c, targets, ordering, accum)

    assert running is not None
    assert accum.float().norm() > 0
    relative = (accum.float() - running).norm() / running.norm()
    assert relative < 5e-3, relative


def test_buffer_carries_the_gradient_of_a_non_autograd_operand():
    """The classifier operand need not carry autograd -- the buffer is what
    asks for dC to be computed. It is not what makes a backward happen: some
    input still has to require grad, which here is the embeddings."""
    e, c, targets = _operands(128, 512, 64, 23)
    assert not c.requires_grad
    accum = torch.zeros_like(c)
    _, de, dc = _run(e, c, targets, None, accum)
    assert dc is None
    assert de is not None
    assert accum.float().abs().sum() > 0


def test_buffer_is_checked_against_the_operands():
    e, c, targets = _operands(128, 512, 64, 31)
    with pytest.raises(ValueError, match="shape"):
        _run(e, c, targets, None, torch.zeros_like(c[:-1]))
    with pytest.raises(ValueError, match="dtype"):
        _run(e, c, targets, None, torch.zeros_like(c, dtype=torch.float32))
    with pytest.raises(ValueError, match="autograd history"):
        _run(e, c, targets, None, torch.zeros_like(c).requires_grad_())
    # An alias of the classifier would be overwritten while it is being read.
    with pytest.raises(ValueError, match="share storage"):
        linear_cross_entropy_apply(
            e.clone().requires_grad_(), c, None, _params(targets, accum=c)
        )
