# Copyright (C) 2024 Apple Inc. All Rights Reserved.
from dataclasses import dataclass
from typing import cast

import torch
import torch.amp

from cut_cross_entropy.cce_backward import cce_backward_kernel
from cut_cross_entropy.cce_lse_forward import cce_lse_forward_kernel
from cut_cross_entropy.constants import IGNORE_INDEX
from cut_cross_entropy.doc import CCE_OPTS_DOC, LINEAR_CROSS_ENTROPY_DOC, add_doc_start
from cut_cross_entropy.tl_autotune import cce_fixed_block_shape
from cut_cross_entropy.utils import (
    TensorInfo,
    _build_flat_valids,
    _handle_eps,
    handle_reduction_none,
)
from cut_cross_entropy.vocab_parallel.utils import (
    VocabParallelOptions,
    vp_reduce_correct_logit,
    vp_reduce_lse,
)


@dataclass
class CCEParams:
    targets: torch.Tensor
    valids: torch.Tensor | None
    softcap: float | None
    reduction: str
    filter_eps: float | None
    shift: int
    batch_shape: torch.Size
    accum_e_fp32: bool
    accum_c_fp32: bool
    filter_e_grad: bool
    filter_c_grad: bool
    vocab_parallel_options: VocabParallelOptions | None
    return_lse: bool
    ## Static vocabulary ordering.
    # An int32 [V] permutation of the classifier rows.  When it is given, both
    # halves tile the vocabulary in this order -- replacing the backward's live
    # argsort of the batch's mean logit -- and the forward additionally stores a
    # per-(vocab tile, row) max logit that lets the backward drop a filtered
    # tile before its recompute matmul instead of after it.  It is a scheduling
    # hint: an ordering that clusters the filtered columns badly costs skipped
    # tiles, never correctness.
    vocab_ordering: torch.Tensor | None = None
    # Optional int32 [n_b_tiles, n_v_tiles] debug output: 1 computed, 0 skipped
    # before the recompute, 2 skipped by the filter after it.
    tile_flags: torch.Tensor | None = None
    # Diagnostic: False forces the late-filter-only path over the same tile
    # grid, so that a caller can check the two decide identically.
    skip_early: bool = True
    # Optional persistent FP32 classifier-gradient accumulator. Backward adds
    # into it without clearing and returns no classifier autograd gradient.
    # It is owned by the caller and deliberately not saved as a version-checked
    # tensor because several sequential backwards may share one accumulator.
    classifier_grad_sink: torch.Tensor | None = None
    # Stripe vocabulary tiles over separate embedding-gradient accumulators,
    # then reduce them once in FP32. One retains the original accumulation.
    e_grad_partitions: int = 1


def _check_vocab_ordering(
    vocab_ordering: torch.Tensor, e: torch.Tensor, c: torch.Tensor, params: "CCEParams"
) -> None:
    """Validate the caller's permutation and the configuration it requires."""
    if vocab_ordering.ndim != 1 or vocab_ordering.numel() != c.size(0):
        raise ValueError(
            f"vocab_ordering must be a 1-D permutation of the {c.size(0)} classifier rows, "
            f"got shape {tuple(vocab_ordering.shape)}."
        )
    if vocab_ordering.dtype != torch.int32:
        raise ValueError(f"vocab_ordering must be int32, got {vocab_ordering.dtype}.")
    if vocab_ordering.stride(0) != 1:
        raise ValueError("vocab_ordering must be contiguous.")
    if params.softcap is not None:
        raise ValueError(
            "vocab_ordering is not supported with softcap: the backward softcaps at the "
            "classifier's width and the forward at fp32, so the stored row max would not "
            "bound the value the backward recomputes."
        )
    if params.vocab_parallel_options is not None:
        raise ValueError(
            "vocab_ordering is not supported with vocab parallelism: each rank holds only a "
            "slice of the classifier, so a global permutation does not tile it."
        )
    if cce_fixed_block_shape(e.dtype) is None:
        raise ValueError(
            "vocab_ordering requires the fixed (non-autotuned) block shape, so that the "
            "forward and the backward walk the same tile grid. Unset CCE_AUTOTUNE."
        )


@torch.compile(fullgraph=True)
def sort_logit_avg(logit_avg: torch.Tensor) -> torch.Tensor:
    return torch.argsort(logit_avg).to(torch.int32)


class LinearCrossEntropyFunction(torch.autograd.Function):
    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(
        ctx,
        e: torch.Tensor,
        c: torch.Tensor,
        bias: torch.Tensor | None,
        params: CCEParams,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        needs_grad = e.requires_grad or c.requires_grad
        if params.classifier_grad_sink is not None and not c.requires_grad:
            raise ValueError("A classifier gradient sink requires a differentiable classifier")
        if bias is not None:
            needs_grad = needs_grad or bias.requires_grad

        filtering = (
            needs_grad
            and params.filter_eps is not None
            and (params.filter_c_grad or params.filter_e_grad)
        )
        vocab_ordering = params.vocab_ordering
        if vocab_ordering is not None:
            _check_vocab_ordering(vocab_ordering, e, c, params)
        # The row max bounds every non-target column of its tile, which is a
        # filtering decision only when both gradients are filtered; with one
        # filter the backward keeps the tile anyway and the store is wasted.
        return_row_max = (
            vocab_ordering is not None
            and filtering
            and params.filter_e_grad
            and params.filter_c_grad
            and e.requires_grad
            and c.requires_grad
        )
        # A caller-supplied ordering replaces the live one, and the backward's
        # argsort disappears with the mean logit that fed it.
        return_logit_avg = filtering and vocab_ordering is None

        e_info = TensorInfo(e.dtype, e.requires_grad)
        c_info = TensorInfo(c.dtype, c.requires_grad)

        bias_info = None
        if bias is not None:
            bias_info = TensorInfo(bias.dtype, bias.requires_grad)

        if torch.is_autocast_enabled():
            e = e.to(dtype=torch.get_autocast_gpu_dtype())
            c = c.to(dtype=torch.get_autocast_gpu_dtype())

            if bias is not None:
                bias = bias.to(dtype=torch.get_autocast_gpu_dtype())

        targets = params.targets
        if (vp_opts := params.vocab_parallel_options) is not None:
            is_my_target = (targets >= vp_opts.start) & (targets < vp_opts.stop)
            targets = torch.where(
                is_my_target,
                targets - vp_opts.start,
                ## NB
                # The backward kernel already uses
                # c.size(0) + 1 as the padding value to ensure that
                # (targets.size(0) % block_size) == 0, so for targets
                # that aren't in this VP rank's range, we can just consider
                # them as padded and all work work as expected.
                targets.new_full((), c.size(0) + 1),
            )

        ret = cce_lse_forward_kernel(
            e=e,
            c=c,
            bias=bias,
            valids=params.valids,
            softcap=params.softcap,
            return_logit_avg=return_logit_avg,
            return_row_max=return_row_max,
            vocab_ordering=vocab_ordering,
            shift=params.shift,
            targets=targets,
        )
        lse = ret.lse
        assert ret.neg_correct_logit is not None
        neg_correct_logit = ret.neg_correct_logit
        logit_avg = ret.logit_avg

        if params.vocab_parallel_options is not None:
            lse = vp_reduce_lse(lse, pg=params.vocab_parallel_options.group)

            neg_correct_logit = vp_reduce_correct_logit(
                neg_correct_logit, pg=params.vocab_parallel_options.group, dtype=lse.dtype
            )

        if ret.row_max is None:
            nll = neg_correct_logit.add_(lse)
        else:
            # The backward reads the target logit out of it; keep it intact.
            nll = neg_correct_logit + lse

        ctx.save_for_backward(
            e,
            c,
            bias,
            lse,
            params.targets,
            params.valids,
            logit_avg,
            ret.row_max,
            neg_correct_logit if ret.row_max is not None else None,
            ret.target_tile,
        )
        ctx.params = params
        ctx.e_info = e_info
        ctx.c_info = c_info
        ctx.bias_info = bias_info

        if not params.return_lse:
            ret_lse = None
        else:
            ret_lse = handle_reduction_none(params.batch_shape, params.valids, params.shift, lse)

        reduction = params.reduction
        if reduction == "mean":
            loss = nll.mean()
        elif reduction == "sum":
            loss = nll.sum()
        elif reduction == "none":
            loss = handle_reduction_none(params.batch_shape, params.valids, params.shift, nll)
        else:
            raise ValueError(f"Unknown reduction {reduction}")

        return loss, ret_lse

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(
        ctx, grad_out: torch.Tensor, grad_lse_out: torch.Tensor | None
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, None]:
        (
            e,
            c,
            bias,
            lse,
            targets,
            valids,
            logit_avg,
            row_max,
            neg_correct_logit,
            target_tile,
        ) = ctx.saved_tensors
        params = cast(CCEParams, ctx.params)

        if params.vocab_ordering is not None:
            # The forward already tiled the vocabulary this way, and the row max
            # it stored bounds only that tiling.
            vocab_ordering = params.vocab_ordering
        elif logit_avg is not None:
            vocab_ordering = sort_logit_avg(logit_avg)
        else:
            vocab_ordering = None

        if not params.skip_early:
            # Diagnostic: the same tile grid, decided by the late filter alone.
            row_max = None
            neg_correct_logit = None

        reduction = params.reduction
        if reduction == "mean":
            grad_scale = 1 / max(lse.numel(), 1)
        elif reduction == "sum":
            grad_scale = 1.0
        elif reduction == "none":
            grad_scale = 1.0
            grad_out = grad_out.view(-1)
        else:
            raise ValueError(f"Unknown reduction {reduction}")

        if grad_lse_out is not None:
            grad_lse_out = grad_lse_out.view(-1)

        reduce_e_grad = False
        pg = None
        if (vp_opts := params.vocab_parallel_options) is not None:
            is_my_target = (targets >= vp_opts.start) & (targets < vp_opts.stop)
            targets = torch.where(
                is_my_target,
                targets - vp_opts.start,
                ## NB
                # The backward kernel already uses
                # c.size(0) + 1 as the padding value to ensure that
                # (targets.size(0) % block_size) == 0, so for targets
                # that aren't in this VP rank's range, we can just consider
                # them as padded and all work work as expected.
                targets.new_full((), c.size(0) + 1),
            )

            reduce_e_grad = vp_opts.reduce_e_grad
            pg = vp_opts.group

        de, dc, dbias = cce_backward_kernel(
            do=grad_out,
            dlse=grad_lse_out,
            e=e,
            e_info=ctx.e_info,
            c=c,
            c_info=ctx.c_info,
            bias=bias,
            bias_info=ctx.bias_info,
            lse=lse,
            valids=valids,
            softcap=params.softcap,
            filter_eps=params.filter_eps,
            targets=targets,
            shift=params.shift,
            vocab_ordering=vocab_ordering,
            row_max=row_max,
            neg_correct_logit=neg_correct_logit,
            tile_flags=params.tile_flags,
            grad_scale=grad_scale,
            accum_e_fp32=params.accum_e_fp32,
            accum_c_fp32=params.accum_c_fp32,
            filter_e_grad=params.filter_e_grad,
            filter_c_grad=params.filter_c_grad,
            reduce_e_grad=reduce_e_grad,
            pg=pg,
            target_tile=target_tile,
            classifier_grad_sink=params.classifier_grad_sink,
            e_grad_partitions=params.e_grad_partitions,
        )

        return de, dc, dbias, None


def linear_cross_entropy_apply(
    e: torch.Tensor,
    c: torch.Tensor,
    bias: torch.Tensor | None,
    params: CCEParams,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    loss, lse = cast(
        tuple[torch.Tensor, torch.Tensor | None],
        LinearCrossEntropyFunction.apply(e, c, bias, params),
    )

    if params.shift != 0 and params.reduction == "none":
        loss = loss[..., params.shift :]

    if params.return_lse and params.shift != 0:
        assert lse is not None
        lse = lse[..., params.shift :]

    return loss, lse


@add_doc_start(LINEAR_CROSS_ENTROPY_DOC)
@add_doc_start(*(doc_str + "\n" for doc_str in CCE_OPTS_DOC))
def cce_linear_cross_entropy(
    e: torch.Tensor,
    c: torch.Tensor,
    targets: torch.Tensor,
    bias: torch.Tensor | None = None,
    ignore_index: int = IGNORE_INDEX,
    softcap: float | None = None,
    reduction: str = "mean",
    shift: bool | int = 0,
    return_lse: bool = False,
    filter_eps: float | str | None = "auto",
    accum_e_fp32: bool = False,
    accum_c_fp32: bool = False,
    filter_e_grad: bool = True,
    filter_c_grad: bool = True,
    vocab_parallel_options: VocabParallelOptions | None = None,
    vocab_ordering: torch.Tensor | None = None,
    tile_flags: torch.Tensor | None = None,
    skip_early: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    assert e.size()[0:-1] == targets.size()
    assert e.size(-1) == c.size(1)
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError(
            "Cut Cross Entropy requires an ampere GPU or newer. "
            "Consider using torch_compile_linear_cross_entropy for scenarios where one is not available."
        )

    batch_shape = targets.size()

    e = e.contiguous()
    targets = targets.contiguous()

    shift = int(shift)
    valids = _build_flat_valids(targets, ignore_index, shift)

    e = e.flatten(0, -2)
    targets = targets.flatten()

    if (targets.data_ptr() % 16) != 0:
        targets = torch.nn.functional.pad(targets, (0, 1))[:-1]

    assert (targets.data_ptr() % 16) == 0
    cce_params = CCEParams(
        targets,
        valids,
        softcap,
        reduction,
        _handle_eps(
            filter_eps, torch.get_autocast_gpu_dtype() if torch.is_autocast_enabled() else e.dtype
        ),
        shift,
        batch_shape,
        accum_e_fp32,
        accum_c_fp32,
        filter_e_grad=filter_e_grad and filter_eps is not None,
        filter_c_grad=filter_c_grad and filter_eps is not None,
        vocab_parallel_options=vocab_parallel_options,
        return_lse=return_lse,
        vocab_ordering=vocab_ordering,
        tile_flags=tile_flags,
        skip_early=skip_early,
    )

    return linear_cross_entropy_apply(e, c, bias, cce_params)
