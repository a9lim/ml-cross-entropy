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
    "baseline": (False, False, False),
    "metadata": (True, False, False),
    "sink": (True, True, False),
    "e-atomic": (True, False, True),
    "sink-e-atomic": (True, True, True),
}


def difference(a, b):
    delta = (a - b).norm()
    norm = b.norm().clamp_min(1e-30)
    return {"relative_l2": (delta / norm).item(), "norm_ratio": (a.norm() / norm).item()}


def compiled_resources():
    from cut_cross_entropy.cce_backward import _cce_backward_kernel
    from cut_cross_entropy.cce_lse_forward import _cce_lse_forward_kernel

    resources = []
    for fn in (_cce_lse_forward_kernel, _cce_backward_kernel):
        while not hasattr(fn, "device_caches"):
            fn = fn.fn
        for kernel in fn.device_caches[torch.cuda.current_device()][0].values():
            entry = {
                "kernel": kernel.name,
                "registers": kernel.n_regs,
                "spills": kernel.n_spills,
                "shared_bytes": kernel.metadata.shared,
                "warps": kernel.metadata.num_warps,
                "stages": kernel.metadata.num_stages,
            }
            if entry not in resources:
                resources.append(entry)
    return resources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input")
    parser.add_argument("--modes", default="baseline,metadata")
    parser.add_argument("--tokens", type=int, default=4096)
    parser.add_argument("--vocab", type=int, default=151936)
    parser.add_argument("--dim", type=int, default=768)
    parser.add_argument("--z-coef", type=float, default=1e-4)
    parser.add_argument("--replays", type=int, default=30)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--warmup-replays", type=int, default=20)
    parser.add_argument("--stages", type=int, choices=[2, 3, 4, 5])
    parser.add_argument("--warps", type=int, choices=[4, 8])
    parser.add_argument("--block-b", type=int, choices=[32, 64, 128])
    parser.add_argument("--block-v", type=int, choices=[32, 64, 128, 256])
    parser.add_argument("--reference-gradients")
    parser.add_argument("--save-gradients")
    args = parser.parse_args()
    if any(value is not None for value in (args.stages, args.warps, args.block_b, args.block_v)):
        # Benchmark-only overrides: keep the forward/backward instruction and
        # tile configurations paired, without changing installed defaults.
        from cut_cross_entropy import tl_autotune
        from cut_cross_entropy.cce_backward import _cce_backward_kernel
        from cut_cross_entropy.cce_lse_forward import _cce_lse_forward_kernel

        config = tl_autotune._cce_best_config()
        block = dict(config.kwargs)
        if args.block_b is not None:
            block["BLOCK_B"] = args.block_b
        if args.block_v is not None:
            block["BLOCK_V"] = args.block_v
        config = type(config)(
            block,
            num_warps=args.warps or config.num_warps,
            num_stages=args.stages or config.num_stages,
        )
        # Forward metadata and backward validation must see exactly the same
        # tile geometry as both compiled kernels, not the installed defaults.
        tl_autotune._cce_best_config = lambda: config

        for kernel in (_cce_lse_forward_kernel, _cce_backward_kernel):
            if not hasattr(kernel, "values"):
                raise RuntimeError("Paired overrides require CCE_AUTOTUNE=0")
            for key, value in config.all_kwargs().items():
                kernel.values[key] = lambda _, value=value: value
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
                "stages_override": args.stages,
                "warps_override": args.warps,
                "block_b_override": args.block_b,
                "block_v_override": args.block_v,
                "note": "baseline disables metadata reads but retains new forward metadata writes",
            }
        ),
        flush=True,
    )

    def run(mode):
        use_metadata, use_sink, embedding_atomic = MODES[mode]
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
                embedding_atomic=embedding_atomic,
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
        # Replay does not allocate through PyTorch, so capture's transient
        # allocations must be included in the allocator peak measurement.
        torch.cuda.reset_peak_memory_stats()
        before_capture_mib = torch.cuda.memory_allocated() / 2**20
        with torch.cuda.graph(graph, stream=stream):
            outputs = body()
        for _ in range(args.warmup_replays):
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
        # Diagnose the changed tile filter outside timing. Every token/vocab
        # pair belongs to one tile; tile counts alone are not comparable across
        # geometries, so report the kept fraction and grid as well.
        from cut_cross_entropy.tl_autotune import cce_fixed_block_shape

        bb, bv = cce_fixed_block_shape(e.dtype)
        flags = torch.empty(
            ((e.shape[0] + bb - 1) // bb, (c.shape[0] + bv - 1) // bv),
            device="cuda",
            dtype=torch.int32,
        )
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
        )
        cce_backward_kernel(
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
            tile_flags=flags,
            grad_scale=1 / e.shape[0],
            classifier_grad_sink=sink if use_sink else None,
            embedding_atomic=embedding_atomic,
        )
        result = {
            "mode": mode,
            "median_ms": statistics.median(timings),
            "round_ms": timings,
            "ce": outputs[0].item(),
            "z_squared": outputs[1].item(),
            "peak_allocated_mib": peak[0],
            "allocated_before_capture_mib": before_capture_mib,
            "capture_peak_delta_mib": peak[0] - before_capture_mib,
            "peak_reserved_mib": peak[1],
            "dE_norm": de.norm().item(),
            "dC_norm": dc.norm().item(),
            "finite_dE": bool(torch.isfinite(de).all()),
            "finite_dC": bool(torch.isfinite(dc).all()),
            "compiled_resources_so_far": compiled_resources(),
            "filter_grid": list(flags.shape),
            "computed_tiles": int((flags == 1).sum().item()),
            "computed_tile_fraction": (flags == 1).float().mean().item(),
        }
        return result, de, dc, (flags == 1).cpu()

    reference = None
    reference_mask = None
    if args.reference_gradients:
        saved = torch.load(args.reference_gradients, map_location="cpu", weights_only=True)
        reference = saved["dE"], saved["dC"]
        reference_mask = saved.get("tile_mask")
        del saved
    first_mode = True
    for mode in args.modes.split(","):
        result, de, dc, tile_mask = run(mode)
        if first_mode and args.save_gradients:
            torch.save({"dE": de, "dC": dc, "tile_mask": tile_mask, "input": metadata, "mode": mode}, args.save_gradients)
        first_mode = False
        if reference is None:
            reference = de, dc
            reference_mask = tile_mask
        else:
            result["dE_vs_first"] = difference(de, reference[0])
            result["dC_vs_first"] = difference(dc, reference[1])
        if reference_mask is not None and reference_mask.shape == tile_mask.shape:
            result["same_filter_mask_as_reference"] = bool(torch.equal(tile_mask, reference_mask))
        print(json.dumps(result), flush=True)
        del de, dc
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
