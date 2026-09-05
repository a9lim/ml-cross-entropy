"""Two-stage LSE numerical and CUDA-capture qualification."""

import pytest
import torch

from cut_cross_entropy.cce_backward import cce_backward_kernel
from cut_cross_entropy.cce_lse_forward import cce_lse_forward_kernel
from cut_cross_entropy.utils import TensorInfo

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Test requires CUDA")


@pytest.mark.parametrize("vocab", [257, 4099])
@pytest.mark.parametrize("nonfinite", [None, "nan", "inf"])
def test_lse_tree_forward_and_gradients(vocab, nonfinite):
    torch.manual_seed(945)
    b, d = 259, 128
    e = torch.randn(b, d, device="cuda", dtype=torch.bfloat16) * 0.125
    c = torch.randn(vocab, d, device="cuda", dtype=torch.bfloat16) * 0.125
    if nonfinite is not None:
        c[17, 0] = float(nonfinite)
    order = torch.randperm(vocab, device="cuda", dtype=torch.int32)
    targets = torch.randint(vocab, (b,), device="cuda")
    results = []
    for reduction in ("atomic", "tree"):
        ret = cce_lse_forward_kernel(
            e,
            c,
            targets=targets,
            vocab_ordering=order,
            return_row_max=True,
            lse_reduction=reduction,
        )
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
        results.append((ret, de, dc))
    old, new = results
    torch.testing.assert_close(new[0].lse, old[0].lse, atol=2e-5, rtol=2e-5, equal_nan=True)
    torch.testing.assert_close(
        new[0].neg_correct_logit, old[0].neg_correct_logit, atol=0, rtol=0, equal_nan=True
    )
    for actual, reference in zip(new[1:], old[1:]):
        if nonfinite is None:
            # FP32 reduction reassociation may move a BF16 probability across
            # one rounding boundary; use the existing BF16 head's 1% envelope.
            assert ((actual - reference).norm() / reference.norm()).item() < 1e-2
            assert torch.isfinite(actual).all()
        else:
            torch.testing.assert_close(torch.isfinite(actual), torch.isfinite(reference))


def test_lse_tree_cuda_graph_replay_reads_new_inputs():
    torch.manual_seed(953)
    e = torch.randn(129, 128, device="cuda", dtype=torch.bfloat16) * 0.125
    c = torch.randn(4099, 128, device="cuda", dtype=torch.bfloat16) * 0.125
    targets = torch.randint(c.shape[0], (e.shape[0],), device="cuda")
    order = torch.randperm(c.shape[0], device="cuda", dtype=torch.int32)

    def forward():
        return cce_lse_forward_kernel(
            e,
            c,
            targets=targets,
            vocab_ordering=order,
            return_row_max=True,
            lse_reduction="tree",
        )

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        forward()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        captured = forward()
    e.mul_(2)
    graph.replay()
    eager = forward()
    torch.testing.assert_close(captured.lse, eager.lse, rtol=0, atol=0)
    torch.testing.assert_close(captured.target_tile, eager.target_tile, rtol=0, atol=0)
