#!/usr/bin/env python3
"""Cross-check graph.json + weights.bin against the .onnx they were exported from."""

import argparse
import json
import sys
import zlib
from math import prod
from pathlib import Path

import onnx
from onnx import AttributeProto, numpy_helper

SCHEMA_VERSION = 1
ALIGN = 64
DTYPE_NBYTES = {"f32": 4}


def attr_value(a):
    """Unwrap an ONNX attribute, re-derived here rather than imported from the exporter."""
    if a.type == AttributeProto.INT:
        return int(a.i)
    if a.type == AttributeProto.FLOAT:
        return float(a.f)
    if a.type == AttributeProto.INTS:
        return [int(v) for v in a.ints]
    if a.type == AttributeProto.FLOATS:
        return [float(v) for v in a.floats]
    if a.type == AttributeProto.STRING:
        return a.s.decode()
    raise AssertionError(f"attribute type {a.type} should never have been exported")


def check_meta(doc, model, problems):
    """Compare schema version and model provenance against the source file."""
    if doc["schema_version"] != SCHEMA_VERSION:
        problems.append(
            f"schema_version is {doc['schema_version']}, this validator knows {SCHEMA_VERSION}")
    opset = next((o.version for o in model.opset_import if o.domain == ""), None)
    if doc["model"]["opset"] != opset:
        problems.append(f"opset: json {doc['model']['opset']}, onnx {opset}")
    if doc["model"]["ir_version"] != model.ir_version:
        problems.append(f"ir_version: json {doc['model']['ir_version']}, onnx {model.ir_version}")


def check_io(doc, graph, problems):
    """Compare graph inputs and outputs, name and shape, in order."""
    init_names = {i.name for i in graph.initializer}
    pairs = (
        ("input", doc["inputs"], [vi for vi in graph.input if vi.name not in init_names]),
        ("output", doc["outputs"], list(graph.output)),
    )
    for label, entries, protos in pairs:
        if len(entries) != len(protos):
            problems.append(f"{label} count: json {len(entries)}, onnx {len(protos)}")
            continue
        for e, vi in zip(entries, protos):
            if e["name"] != vi.name:
                problems.append(f"{label}: json '{e['name']}', onnx '{vi.name}'")
            want = [d.dim_value for d in vi.type.tensor_type.shape.dim]
            if e["shape"] != want:
                problems.append(f"{label} '{vi.name}': shape json {e['shape']}, onnx {want}")


def check_nodes(doc, graph, problems):
    """Compare every node field by field, in file order."""
    got = doc["nodes"]
    if len(got) != len(graph.node):
        problems.append(f"node count: json {len(got)}, onnx {len(graph.node)}")
        return
    for i, (g, nd) in enumerate(zip(got, graph.node)):
        want_name = nd.name or f"{nd.op_type}_{i}"
        where = f"node[{i}] '{want_name}'"
        if g["name"] != want_name:
            problems.append(f"{where}: name in json is '{g['name']}'")
        if g["op_type"] != nd.op_type:
            problems.append(f"{where}: op_type json '{g['op_type']}', onnx '{nd.op_type}'")
        if g["inputs"] != list(nd.input):
            problems.append(f"{where}: inputs json {g['inputs']}, onnx {list(nd.input)}")
        if g["outputs"] != list(nd.output):
            problems.append(f"{where}: outputs json {g['outputs']}, onnx {list(nd.output)}")
        want_attrs = {a.name: attr_value(a) for a in nd.attribute}
        if g["attributes"] != want_attrs:
            problems.append(f"{where}: attributes json {g['attributes']}, onnx {want_attrs}")


def check_weights(doc, blob, graph, problems):
    """Compare the weight blob byte for byte against the tensors in the .onnx."""
    if len(blob) != doc["weights_nbytes"]:
        problems.append(f"weights.bin is {len(blob)} bytes, json says {doc['weights_nbytes']}")

    by_name = {i.name: i for i in graph.initializer}
    entries = doc["initializers"]
    if len(entries) != len(graph.initializer):
        problems.append(f"initializer count: json {len(entries)}, onnx {len(graph.initializer)}")

    ranges = []
    for e in entries:
        name, where = e["name"], f"initializer '{e['name']}'"
        init = by_name.get(name)
        if init is None:
            problems.append(f"{where}: not present in the onnx")
            continue

        if e["shape"] != list(init.dims):
            problems.append(f"{where}: shape json {e['shape']}, onnx {list(init.dims)}")
        if e["offset"] % ALIGN:
            problems.append(f"{where}: offset {e['offset']} is not {ALIGN}-byte aligned")
        if e["nbytes"] != prod(e["shape"]) * DTYPE_NBYTES[e["dtype"]]:
            problems.append(f"{where}: nbytes {e['nbytes']} disagrees with shape {e['shape']}")

        lo, hi = e["offset"], e["offset"] + e["nbytes"]
        if hi > len(blob):
            problems.append(f"{where}: range [{lo},{hi}) runs past the {len(blob)}-byte file")
            continue
        ranges.append((lo, hi, name))

        got = blob[lo:hi]
        if f"{zlib.crc32(got) & 0xFFFFFFFF:08x}" != e["crc32"]:
            problems.append(f"{where}: recorded crc32 does not match the bytes in weights.bin")
        if got != numpy_helper.to_array(init).tobytes():
            problems.append(f"{where}: bytes in weights.bin differ from the onnx tensor")

    ranges.sort()
    for (_, hi, a), (lo2, _, b) in zip(ranges, ranges[1:]):
        if lo2 < hi:
            problems.append(f"initializers '{a}' and '{b}' overlap in weights.bin")

def check_invariants(doc, problems):
    """Check the promises the format itself makes, without consulting the onnx.

    Topological order, one producer per tensor, and no dangling reference. The
    C++ loader is allowed to assume all three, so something has to prove them.
    """
    defined = {v["name"] for v in doc["inputs"]} | {e["name"] for e in doc["initializers"]}
    producer = {}
    for node in doc["nodes"]:
        for t in node["inputs"]:
            if t and t not in defined:
                problems.append(f"node '{node['name']}': input '{t}' is used before it is defined")
        for t in node["outputs"]:
            if t in producer:
                problems.append(
                    f"tensor '{t}': produced by both '{producer[t]}' and '{node['name']}'")
            producer[t] = node["name"]
            defined.add(t)
    for v in doc["outputs"]:
        if v["name"] not in defined:
            problems.append(f"graph output '{v['name']}' is not produced by anything")


def validate(src, out_dir):
    """Run every check and return the problems, empty when the export is trustworthy."""
    model = onnx.load(str(src))
    doc = json.loads((out_dir / "graph.json").read_text())
    blob = (out_dir / doc["weights_file"]).read_bytes()

    problems = []
    check_meta(doc, model, problems)
    check_io(doc, model.graph, problems)
    check_nodes(doc, model.graph, problems)
    check_weights(doc, blob, model.graph, problems)
    check_invariants(doc, problems)
    return doc, problems


def main():
    """CLI: validate one exported model against its source .onnx."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model", type=Path)
    ap.add_argument("-o", "--out", type=Path, default=None)
    args = ap.parse_args()

    out_dir = args.out or Path("artifacts") / args.model.stem
    doc, problems = validate(args.model, out_dir)

    if problems:
        print(f"{out_dir} does NOT match {args.model}:", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        raise SystemExit(1)

    print(f"{out_dir} matches {args.model}: "
          f"{len(doc['nodes'])} nodes, {len(doc['initializers'])} initializers, "
          f"{doc['weights_nbytes']} bytes of weights")


if __name__ == "__main__":
    main()