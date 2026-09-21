#!/usr/bin/env python3
"""Run onnxruntime with every intermediate exposed and write per-layer golden tensors."""

import argparse
import json
import sys
import zlib
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, shape_inference

SCHEMA_VERSION = 1
ALIGN = 64
SEED = 0
REL_TOLERANCE = 1e-4


def make_inputs(graph, seed):
    """
    Build one input tensor per graph input from a fixed seed.

    Normal noise rather than zeros: a zero input makes Relu and MaxPool
    degenerate and would hide exactly the bugs the goldens exist to catch.
    """
    rng = np.random.default_rng(seed)
    init_names = {i.name for i in graph.initializer}
    feeds = {}
    for vi in graph.input:
        if vi.name in init_names:
            continue
        shape = [d.dim_value for d in vi.type.tensor_type.shape.dim]
        feeds[vi.name] = rng.standard_normal(shape, dtype=np.float32)
    return feeds


def instrument(model):
    """
    Add every intermediate tensor to the graph outputs, and return the names added.

    Exposing an intermediate is what stops onnxruntime fusing across it, which
    is why the caller also runs the untouched model and compares.
    """
    model = shape_inference.infer_shapes(model)
    graph = model.graph
    known = {vi.name: vi for vi in graph.value_info}
    already = {o.name for o in graph.output}

    added = []
    for nd in graph.node:
        for name in nd.output:
            if not name or name in already:
                continue
            already.add(name)
            added.append(name)
            graph.output.append(known.get(name) or helper.make_empty_tensor_value_info(name))
    return model, added


def run(model_proto, feeds, disable_optimizations):
    """Run one session and return {output name: array}."""
    opts = ort.SessionOptions()
    if disable_optimizations:
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess = ort.InferenceSession(
        model_proto.SerializeToString(), opts, providers=["CPUExecutionProvider"])
    names = [o.name for o in sess.get_outputs()]
    return dict(zip(names, sess.run(None, feeds)))


def ordered_names(graph, feeds):
    """Graph inputs first, then every node output in node order."""
    names = list(feeds)
    for nd in graph.node:
        names.extend(n for n in nd.output if n)
    return names


def pack(names, arrays):
    """Pack the tensors into one aligned blob plus manifest entries."""
    blob, entries, offset = bytearray(), [], 0
    for name in names:
        raw = np.ascontiguousarray(arrays[name], dtype=np.float32).tobytes()
        gap = (-offset) % ALIGN
        blob.extend(b"\0" * gap)
        offset += gap
        entries.append({
            "name": name,
            "dtype": "f32",
            "shape": list(arrays[name].shape),
            "offset": offset,
            "nbytes": len(raw),
            "crc32": f"{zlib.crc32(raw) & 0xFFFFFFFF:08x}",
        })
        blob.extend(raw)
        offset += len(raw)
    return bytes(blob), entries


def compare_clean(src, feeds, instrumented):
    """
    Re-run the untouched model and compare its outputs against the instrumented run.

    Exposing intermediates disables fusion, so the instrumented graph is not
    the graph onnxruntime would normally execute. If those two disagree by more
    than float noise, the goldens describe a pipeline nobody will ever run.
    """
    clean = run(onnx.load(str(src)), feeds, disable_optimizations=False)
    report = []
    for name, ref in clean.items():
        got = instrumented[name]
        scale = float(np.max(np.abs(ref))) or 1.0
        abs_err = float(np.max(np.abs(got - ref)))
        report.append((name, abs_err, abs_err / scale))
    return report


def dump(src, out_dir, seed):
    """Produce golden.bin + golden_manifest.json for one model."""
    model, added = instrument(onnx.load(str(src)))
    feeds = make_inputs(model.graph, seed)
    arrays = run(model, feeds, disable_optimizations=True)
    arrays.update(feeds)

    names = ordered_names(model.graph, feeds)
    blob, entries = pack(names, arrays)
    clean_report = compare_clean(src, feeds, arrays)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "model": src.stem,
        "source": str(src),
        "seed": seed,
        "onnxruntime": ort.__version__,
        "golden_file": "golden.bin",
        "golden_nbytes": len(blob),
        "tensors": entries,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "golden.bin").write_bytes(blob)
    (out_dir / "golden_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest, added, clean_report


def main():
    """CLI: dump per-layer goldens for one model."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model", type=Path)
    ap.add_argument("-o", "--out", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    out_dir = args.out or Path("artifacts") / args.model.stem
    manifest, added, clean_report = dump(args.model, out_dir, args.seed)

    print(f"{out_dir}/golden.bin  {len(manifest['tensors'])} tensors "
          f"({len(added)} intermediates exposed), {manifest['golden_nbytes'] / 1e6:.1f} MB")

    worst = 0.0
    for name, abs_err, rel_err in clean_report:
        print(f"  clean vs instrumented, '{name}': "
              f"max abs {abs_err:.3e}, relative {rel_err:.3e}")
        worst = max(worst, rel_err)
    if worst > REL_TOLERANCE:
        print(f"final output differs by {worst:.3e} relative, over the {REL_TOLERANCE:.0e} "
              f"tolerance -- the instrumented graph is not computing the same thing",
              file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()