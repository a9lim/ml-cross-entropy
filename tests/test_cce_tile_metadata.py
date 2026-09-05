"""The early filter must keep exactly the late filter's tile set."""

import pytest
import torch
import triton

from cut_cross_entropy.cce_backward import cce_backward_kernel
from cut_cross_entropy.cce_lse_forward import cce_lse_forward_kernel
from cut_cross_entropy.utils import TensorInfo

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Test requires CUDA")


@pytest.mark.parametrize("vocab", [256, 257])
@pytest.mark.parametrize("selected_shifted_rows", [False, True])
@pytest.mark.parametrize("nonfinite", [None, "nan", "inf"])
def test_target_tile_matches_membership_and_filtered_gradients(
    vocab, selected_shifted_rows, nonfinite
):
    torch.manual_seed(932)
    b, d = 131, 64
    e = torch.randn(b, d, device="cuda", dtype=torch.bfloat16) * 0.125
    c = torch.randn(vocab, d, device="cuda", dtype=torch.bfloat16) * 0.125
    if nonfinite is not None:
        c[17, 0] = float(nonfinite)
    order = torch.randperm(vocab, device="cuda", dtype=torch.int32)
    shift = int(selected_shifted_rows)
    targets = torch.randint(vocab, (b + shift,), device="cuda")
    # Retain the low-level kernel's behavior for invalid target ids, including
    # V itself, which matches padded columns only when the final tile is ragged.
    targets[:4] = torch.tensor([-1, vocab, vocab + 1, 0], device="cuda")
    valids = torch.arange(0, b - shift, 2, device="cuda") if selected_shifted_rows else None
    ret = cce_lse_forward_kernel(
        e,
        c,
        targets=targets,
        valids=valids,
        shift=shift,
        vocab_ordering=order,
        return_row_max=True,
    )
    actual_targets = targets[valids + shift] if valids is not None else targets
    padded_order = torch.nn.functional.pad(order, (0, (-vocab) % 128), value=vocab)
    matches = actual_targets[:, None] == padded_order[None, :]
    expected = torch.where(matches.any(1), matches.to(torch.int32).argmax(1) // 128, -1).to(
        torch.int32
    )
    torch.testing.assert_close(ret.target_tile, expected, rtol=0, atol=0)
    rows = ret.lse.numel()
    # LSE is compact over valids, but its incoming derivative uses original
    # shifted row addresses, as the public autograd wrapper returns it.
    dlse = torch.zeros(b + shift, device="cuda")
    if valids is None:
        dlse.copy_(2e-4 * ret.lse / rows)
    else:
        dlse[valids + shift] = 2e-4 * ret.lse / rows
    outputs = []
    for mode in ("late", "legacy_early", "metadata_early"):
        flags = torch.empty(
            (triton.cdiv(rows, 128), triton.cdiv(vocab, 128)), device="cuda", dtype=torch.int32
        )
        de, dc, _ = cce_backward_kernel(
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
            filter_eps=0.01,
            targets=targets,
            shift=shift,
            vocab_ordering=order,
            row_max=ret.row_max if mode != "late" else None,
            neg_correct_logit=ret.neg_correct_logit if mode != "late" else None,
            target_tile=ret.target_tile if mode == "metadata_early" else None,
            tile_flags=flags,
            grad_scale=1 / rows,
            accum_e_fp32=True,
            accum_c_fp32=True,
        )
        outputs.append((flags == 1, de, dc))
    for actual in outputs[1:]:
        torch.testing.assert_close(actual[0], outputs[0][0], rtol=0, atol=0)
        for gradient, reference in zip(actual[1:], outputs[0][1:]):
            torch.testing.assert_close(gradient, reference, rtol=1e-5, atol=1e-6, equal_nan=True)
