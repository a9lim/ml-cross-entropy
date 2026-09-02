# Copyright (C) 2024 Apple Inc. All Rights Reserved.
from dataclasses import dataclass

import torch
import triton
import triton.language as tl

from cut_cross_entropy.tl_autotune import cce_fixed_block_shape, cce_forward_autotune
from cut_cross_entropy.tl_utils import b_bin_fn, tl_logaddexp, tl_softcapping


def _cce_lse_forward_kernel(
    E,
    C,
    Bias,
    LSE,
    LA,
    NegCorrectLogit,
    RowMax,
    Locks,
    Valids,
    Targets,
    VocabOrdering,
    softcap,
    shift,
    B,
    V,
    D,
    BMax,
    stride_eb,
    stride_ed,
    stride_cv,
    stride_cd,
    stride_biasv,
    stride_vb,
    num_locks,
    # Meta-parameters
    B_BIN,
    HAS_BIAS: tl.constexpr,
    HAS_VALIDS: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_V: tl.constexpr,
    BLOCK_D: tl.constexpr,  #
    GROUP_B: tl.constexpr,  #
    EVEN_D: tl.constexpr,
    HAS_SOFTCAP: tl.constexpr,
    HAS_LA: tl.constexpr,
    DOT_PRECISION: tl.constexpr,
    HAS_TARGETS: tl.constexpr,
    HAS_SHIFT: tl.constexpr,
    HAS_ROWMAX: tl.constexpr,
    HAS_VOCAB_ORDERING: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_b = tl.cdiv(B, BLOCK_B)
    num_pid_v = tl.cdiv(V, BLOCK_V)
    num_pid_in_group = GROUP_B * num_pid_v
    group_id = pid // num_pid_in_group
    first_pid_b = group_id * GROUP_B
    group_size_b = min(num_pid_b - first_pid_b, GROUP_B)
    pid_b = (first_pid_b + ((pid % num_pid_in_group) % group_size_b)).to(tl.int64)
    pid_v = ((pid % num_pid_in_group) // group_size_b).to(tl.int64)

    offs_b = (pid_b * BLOCK_B + tl.arange(0, BLOCK_B)).to(tl.int64)
    if HAS_VALIDS:
        offs_b = tl.load(Valids + stride_vb * offs_b, mask=offs_b < B, other=BMax).to(tl.int64)

    offs_v = (pid_v * BLOCK_V + tl.arange(0, BLOCK_V)).to(tl.int64)
    if HAS_VOCAB_ORDERING:
        offs_v = tl.load(VocabOrdering + offs_v, mask=offs_v < V, other=V).to(tl.int64)

    offs_d = tl.arange(0, BLOCK_D).to(tl.int64)
    e_ptrs = E + (offs_b[:, None] * stride_eb + offs_d[None, :] * stride_ed)
    c_ptrs = C + (offs_v[None, :] * stride_cv + offs_d[:, None] * stride_cd)

    accum = tl.zeros((BLOCK_B, BLOCK_V), dtype=tl.float32)
    for d in range(0, tl.cdiv(D, BLOCK_D)):
        e_mask = offs_b[:, None] < BMax
        if not EVEN_D:
            e_mask = e_mask & (offs_d[None, :] < (D - d * BLOCK_D))

        e = tl.load(e_ptrs, mask=e_mask, other=0.0)

        c_mask = offs_v[None, :] < V
        if not EVEN_D:
            c_mask = c_mask & (offs_d[:, None] < (D - d * BLOCK_D))

        c = tl.load(c_ptrs, mask=c_mask, other=0.0)

        accum = tl.dot(e, c, accum, input_precision=DOT_PRECISION)

        e_ptrs += BLOCK_D * stride_ed
        c_ptrs += BLOCK_D * stride_cd

    tl.debug_barrier()

    accum = accum.cast(E.dtype.element_ty, fp_downcast_rounding="rtne")
    if HAS_BIAS:
        bias = tl.load(Bias + offs_v * stride_biasv, mask=offs_v < V, other=0.0)
        accum += bias[None, :]

    logits = tl.where(offs_v[None, :] < V, accum, -float("inf"))
    if HAS_SOFTCAP:
        logits = tl_softcapping(logits, softcap)

    logits = logits.cast(tl.float32)
    if HAS_LA:
        this_avg_logit = tl.sum(logits, 0) / B
        tl.atomic_add(LA + offs_v, this_avg_logit, mask=offs_v < V)

    if HAS_TARGETS:
        if HAS_SHIFT:
            target_offs_b = offs_b + shift
        else:
            target_offs_b = offs_b

        this_targets = tl.load(Targets + target_offs_b, mask=target_offs_b < BMax, other=V + 1)

        offs_b = (pid_b * BLOCK_B + tl.arange(0, BLOCK_B)).to(tl.int64)

        neg_correct_logit_ptrs = NegCorrectLogit + offs_b

        neg_correct_logit_ptrs = tl.broadcast_to(
            neg_correct_logit_ptrs[:, None], (BLOCK_B, BLOCK_V)
        )
        tl.store(neg_correct_logit_ptrs, -logits, mask=this_targets[:, None] == offs_v[None, :])

        if HAS_ROWMAX:
            # The largest logit over this tile's non-target columns.  Those
            # columns contribute d_accum = exp(logit - lse) to the backward's
            # gradient filter, and both the subtraction and exp are monotone,
            # so this single value decides all of them.  Columns past V are
            # already -inf (the tail-tile mask above) and the target column is
            # masked to -inf here, so neither can raise the bound.
            nt_logits = tl.where(this_targets[:, None] == offs_v[None, :], -float("inf"), logits)
            # tl.max does not propagate NaN, and the backward keeps every tile
            # it cannot prove small (its test is written `< filter_eps`), so a
            # non-finite logit must not hide inside a finite bound: send the
            # whole row to the slow path instead.
            nt_logits = tl.where(nt_logits != nt_logits, float("inf"), nt_logits)
            # Stored tile-major so a backward program reads BLOCK_B contiguous
            # floats.
            tl.store(RowMax + pid_v * B + offs_b, tl.max(nt_logits, axis=1), mask=offs_b < B)
    else:
        offs_b = (pid_b * BLOCK_B + tl.arange(0, BLOCK_B)).to(tl.int64)

    this_mx = tl.max(logits, axis=1)
    this_lse = this_mx + tl.log(tl.sum(tl.exp(logits - this_mx[:, None]), axis=1))

    o_mask = offs_b < B

    lse_ptrs = LSE + offs_b

    this_locks = Locks + (pid_b // tl.cdiv(B, BLOCK_B * num_locks))
    while tl.atomic_cas(this_locks, 0, 1) == 1:
        pass

    lse = tl.load(lse_ptrs, mask=o_mask, other=0.0, eviction_policy="evict_last")
    lse = tl_logaddexp(lse, this_lse)
    lse = tl.store(lse_ptrs, lse, mask=o_mask, eviction_policy="evict_last")

    tl.debug_barrier()
    tl.atomic_xchg(this_locks, 0)


_cce_lse_forward_kernel = triton.jit(_cce_lse_forward_kernel)
_cce_lse_forward_kernel = triton.heuristics(  # type: ignore
    {
        "EVEN_D": lambda args: args["D"] % args["BLOCK_D"] == 0,
        "HAS_BIAS": lambda args: args["Bias"] is not None,
        "HAS_VALIDS": lambda args: args["Valids"] is not None,
        "HAS_SOFTCAP": lambda args: args["softcap"] is not None,
        "HAS_LA": lambda args: args["LA"] is not None,
        "GROUP_B": lambda args: 8,
        "DOT_PRECISION": lambda args: "tf32"
        if torch.get_float32_matmul_precision() == "high"
        else "ieee",
        "HAS_TARGETS": lambda args: args["Targets"] is not None,
        "HAS_SHIFT": lambda args: args["shift"] != 0,
        "HAS_ROWMAX": lambda args: args["RowMax"] is not None,
        "HAS_VOCAB_ORDERING": lambda args: args["VocabOrdering"] is not None,
    }
)(_cce_lse_forward_kernel)
_cce_lse_forward_kernel = cce_forward_autotune()(_cce_lse_forward_kernel)  # type: ignore


@dataclass(slots=True)
class LSEReturn:
    lse: torch.Tensor
    logit_avg: torch.Tensor | None
    neg_correct_logit: torch.Tensor | None
    row_max: torch.Tensor | None


def cce_lse_forward_kernel(
    e: torch.Tensor,
    c: torch.Tensor,
    bias: torch.Tensor | None = None,
    valids: torch.Tensor | None = None,
    softcap: float | None = None,
    targets: torch.Tensor | None = None,
    shift: int = 0,
    return_logit_avg: bool = False,
    return_row_max: bool = False,
    vocab_ordering: torch.Tensor | None = None,
) -> LSEReturn:
    """Compute the per-row LSE, and optionally the classifier's mean logit.

    ``vocab_ordering`` is an int32 permutation of the classifier rows; when it
    is given, the vocabulary is tiled in that order.  ``return_row_max`` then
    also stores, per (vocab tile, row), the largest logit over that tile's
    non-target columns, which is what lets the backward decide a tile's
    gradient filter before recomputing its logits.
    """
    # Check constraints.
    assert e.shape[1] == c.shape[1], "Incompatible dimensions"
    assert e.is_contiguous(), "Matrix A must be contiguous"
    if valids is not None:
        assert valids.ndim == 1
        B = valids.numel()
    else:
        B, _ = e.shape

    if bias is not None:
        assert bias.ndim == 1
        assert c.shape[0] == bias.shape[0]

    V, D = c.shape
    # Allocates output.
    lse = e.new_full((B,), -torch.inf, dtype=torch.float32)
    neg_correct_logit = e.new_full((B,), 0.0, dtype=torch.float32) if targets is not None else None
    assert lse.stride(0) == 1

    locks = e.new_full(
        (triton.cdiv(B, 128),),
        0,
        dtype=torch.uint32,
    )

    if return_logit_avg:
        logit_avg = e.new_full((V,), 0.0, dtype=torch.float32)
    else:
        logit_avg = None

    if vocab_ordering is not None:
        assert vocab_ordering.ndim == 1
        assert vocab_ordering.numel() == c.size(0)
        assert vocab_ordering.dtype == torch.int32
        assert vocab_ordering.stride(0) == 1

    if return_row_max:
        assert vocab_ordering is not None, "the row max is only valid for a known vocab tiling"
        assert targets is not None, "the row max excludes the target column"
        assert softcap is None, "softcap is applied at a different width in the backward"
        block_shape = cce_fixed_block_shape(e.dtype)
        assert block_shape is not None, "the row max needs the fixed (non-autotuned) config"
        # [vocab tile, row]: every element is written by exactly one program,
        # before any backward program reads it, so it needs no initialization.
        row_max = e.new_empty((triton.cdiv(V, block_shape[1]), B), dtype=torch.float32)
    else:
        row_max = None

    # 1D launch kernel where each block gets its own program.
    def grid(META) -> tuple[int]:
        return (triton.cdiv(B, META["BLOCK_B"]) * triton.cdiv(V, META["BLOCK_V"]),)

    _cce_lse_forward_kernel[grid](
        e,
        c,
        bias,
        lse,
        logit_avg,
        neg_correct_logit,
        row_max,
        locks,
        valids,
        targets,
        vocab_ordering,
        softcap,
        shift,
        B,
        V,
        D,  #
        e.size(0),
        e.stride(0),
        e.stride(1),  #
        c.stride(0),
        c.stride(1),  #
        1 if bias is None else bias.stride(0),
        1 if valids is None else valids.stride(0),
        num_locks=locks.size(0),
        B_BIN=b_bin_fn(B),
    )

    return LSEReturn(lse, logit_avg, neg_correct_logit, row_max)
