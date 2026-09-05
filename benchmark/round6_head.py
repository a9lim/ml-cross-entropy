"""Paired captured CCE experiments; accepts CPU tensors from a trained head.

python -m benchmark.round6_head --input /tmp/trained_head.pt
Input is {'e': [B,D], 'c': [V,D], 'targets': [B], 'metadata': optional dict}.
Random inputs are a kernel smoke test, not a trained-head speed prediction.
"""

import argparse
import gc
import json
import statistics

import torch

from cut_cross_entropy.cce_backward import cce_backward_kernel
from cut_cross_entropy.cce_lse_forward import cce_lse_forward_kernel
from cut_cross_entropy.utils import TensorInfo, _handle_eps

MODES = {
    "baseline": ("atomic", False, False),
    "metadata": ("atomic", True, False),
    "tree": ("tree", True, False),
    "sink": ("tree", True, True),
    "sink-atomic": ("atomic", True, True),
}


def difference(a, b):
    delta = (a - b).norm()
    norm = b.norm().clamp_min(1e-30)
    return {"relative_l2": (delta / norm).item(), "norm_ratio": (a.norm() / norm).item()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input")
    parser.add_argument("--modes", default="baseline,metadata,tree,sink")
    parser.add_argument("--tokens", type=int, default=4096)
    parser.add_argument("--vocab", type=int, default=151936)
    parser.add_argument("--dim", type=int, default=768)
    parser.add_argument("--z-coef", type=float, default=1e-4)
    parser.add_argument("--replays", type=int, default=30)
    parser.add_argument("--rounds", type=int, default=5)
    args = parser.parse_args()
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(1011)
    if args.input:
        blob = torch.load(args.input, map_location="cpu", weights_only=True)
        e = blob["e"].to(device="cuda", dtype=torch.bfloat16).flatten(0, -2).contiguous()
        c = blob["c"].to(device="cuda", dtype=torch.bfloat16).contiguous()
        targets = blob["targets"].to(device="cuda", dtype=torch.long).flatten().contiguous()
        metadata = blob.get("metadata", {})
        del blob
    else:
        e = torch.randn(args.tokens, args.dim, device="cuda", dtype=torch.bfloat16) * 0.125
        c = torch.randn(args.vocab, args.dim, device="cuda", dtype=torch.bfloat16) * 0.125
        targets = torch.randint(args.vocab, (args.tokens,), device="cuda")
        metadata = {"warning": "random initialization only"}
    if targets.data_ptr() % 16:
        targets = torch.nn.functional.pad(targets, (0, 1))[:-1]
    print(
        json.dumps(
            {
                "input": metadata,
                "shape": [e.shape[0], c.shape[0], e.shape[1]],
                "gpu": torch.cuda.get_device_name(),
                "z_coef": args.z_coef,
                "note": "baseline disables metadata reads but retains new forward metadata writes",
            }
        ),
        flush=True,
    )

    def run(mode):
        reduction, use_metadata, use_sink = MODES[mode]
        sink = torch.zeros(c.shape, device="cuda", dtype=torch.float32)
        scalar = torch.ones((), device="cuda")
        z_coef = torch.tensor(args.z_coef, device="cuda")

        def body():
            mean = e.float().mean(0, keepdim=True)
            averages = torch.addmm(
                torch.zeros(1, c.shape[0], device="cuda"),
                mean.to(c.dtype),
                c.mT,
                out_dtype=torch.float32,
            )
            order = torch.argsort(averages[0], stable=True).to(torch.int32)
            ret = cce_lse_forward_kernel(
                e,
                c,
                targets=targets,
                vocab_ordering=order,
                return_row_max=True,
                lse_reduction=reduction,
            )
            ce = (ret.lse + ret.neg_correct_logit).mean()
            z = ret.lse.square().mean()
            de, dc, _ = cce_backward_kernel(
                do=scalar,
                dlse=2 * z_coef * ret.lse / e.shape[0],
                e=e,
                e_info=TensorInfo(e.dtype, True),
                c=c,
                c_info=TensorInfo(c.dtype, True),
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
                target_tile=ret.target_tile if use_metadata else None,
                grad_scale=1 / e.shape[0],
                classifier_grad_sink=sink if use_sink else None,
            )
            if not use_sink:
                sink.add_(dc)
            return ce, z, de

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                body()
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        sink.zero_()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            outputs = body()
        torch.cuda.reset_peak_memory_stats()
        for _ in range(5):
            graph.replay()
        timings = []
        for _ in range(args.rounds):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(args.replays):
                graph.replay()
            end.record()
            end.synchronize()
            timings.append(start.elapsed_time(end) / args.replays)
        peak = [torch.cuda.max_memory_allocated() / 2**20, torch.cuda.max_memory_reserved() / 2**20]
        sink.zero_()
        graph.replay()
        torch.cuda.synchronize()
        de = outputs[2].float().cpu()
        dc = sink.cpu()
        result = {
            "mode": mode,
            "median_ms": statistics.median(timings),
            "round_ms": timings,
            "ce": outputs[0].item(),
            "z_squared": outputs[1].item(),
            "peak_allocated_mib": peak[0],
            "peak_reserved_mib": peak[1],
            "dE_norm": de.norm().item(),
            "dC_norm": dc.norm().item(),
            "finite_dE": bool(torch.isfinite(de).all()),
            "finite_dC": bool(torch.isfinite(dc).all()),
        }
        return result, de, dc

    reference = None
    for mode in args.modes.split(","):
        result, de, dc = run(mode)
        if reference is None:
            reference = de, dc
        else:
            result["dE_vs_first"] = difference(de, reference[0])
            result["dC_vs_first"] = difference(dc, reference[1])
        print(json.dumps(result), flush=True)
        del de, dc
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
