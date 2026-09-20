#!/usr/bin/env python3
"""ONNX -> graph.json + weights.bin. The brigde between the Python frontend nad the C++ compiler"""

import argparse
import json
import re
import sys
import zlib
from pathlib import Path

import onnx
from onnx import AttributeProto, TensorProto, numpy_helper

SCHEMA_VERSION = 1
ALIGN = 64

SUPPORTED_OPS = {
    "Add", "BatchNormalization", "Conv", "Flatten", "Gemm",
    "GlobalAveragePool", "Identify", "MaxPool", "Relu",
}

class Rejected(Exception):
    """The model violates a v1 constraint; the message lists every violation found."""
    pass

def check_float(elem_type, where, problems):
    """Record a violation if the type is not FLOAT.
    
    Does not raise, so one run reports every problem instead of the first one.
    """
    if elem_type != TensorProto.FLOAT:
        got = TensorProto.DataType.Name(elem_type)
        problems.append(f"{where}: dtype {got}, v1 supports only FLOAT")


def value_info(vi, problems):
    """
    Translate an ONNX graph input or output into a schema entry.

    A dynamic dimension is a violation - v1 is strictly static.
    """
    dims = []
    for d in vi.type.tensor_type.shape.dim:
        if d.HasField("dim_value"):
            dims.append(d.dim_value)
        else:
            problems.append(f"tensor '{vi.name}': dynamic dimension '{d.dim_param or '?'}'")
    check_float(vi.type.tensor_type.elem_type, f"tensor '{vi.name}'", problems)
    return {"name": vi.name, "dtype": "f32", "shape": dims}

def attribute(a, node_name, problems):
    """
    Unwrap an ONNX attribute into a plain Python value.
    
    Only the types our 9 operators use are handled. Anything else (TENSOR, GRAPH, SPARSE_TENSOR) is a violation.
    """
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
    kind = AttributeProto.AttributeType.Name(a.type)
    problems.append(f"node '{node_name}': attribute '{a.name}' has type {kind}, unsupported")
    return None

def check_limits(name, op_type, attrs, problems):
    """
    Check the v1 restrictions for one operator.
    
    Deliberately narrow: grouped and dilated convolutions are rejected rather than implemented, becouse neither frozen model uses them.
    """
    if op_type not in SUPPORTED_OPS:
        problems.append(f"node '{name}': operator '{op_type}' is not in the v1 op set")
        return
    if op_type == "Conv":
        if attrs.get("group", 1) != 1:
            problems.append(f"node '{name}': group={attrs['group']}, v1 supports 1 only")
        if attrs.get("dilations", [1, 1]) != [1, 1]:
            problems.append(f"node '{name}': dilations = {attrs['dilations']}, v1 supports [1,1] only")

def pack_weights(graph, problems):
    """
    Pack every initializer into one blob, each aligned to ALIGN bytes.
    
    Returns (bytes, entries), where an entry carries the offset, length and crc32 of one tensor. Alignment is for vectorization in Phase 4. The crc32
    is what the validator uses to compare the blob against the original .onnx
    """
    blob, entries, offset = bytearray(), [], 0
    for init in graph.initializer:
        check_float(init.data_type, f"initializer '{init.name}'", problems)
        raw = numpy_helper.to_array(init).tobytes()
        gap = (-offset) % ALIGN
        blob.extend(b"\0" * gap)
        offset += gap
        entries.append({
            "name": init.name,
            "dtype": "f32",
            "shape": list(init.dims),
            "offset": offset,
            "nbytes": len(raw),
            "crc32": f"{zlib.crc32(raw) & 0xFFFFFFFF:08x}",
        })
        blob.extend(raw)
        offset += len(raw)
    return bytes(blob), entries

def build_nodes(graph, problems):
    """
    Translate ONNX nodes into schema entries, preserving the order in the file.

    Unnamed nodes get a name from their index so every error message can point at a 
    specific node. Intermediate shapes are deliberately omitted - shape inference is Phase 1's job.
    """
    nodes, seen = [], set()
    for i, nd in enumerate(graph.node):
        name = nd.name or f"{nd.op_type}_{i}"
        if name in seen:
            problems.append(f"node '{name}': duplicate name")
        seen.add(name)

        attrs = {a.name: attribute(a, name, problems) for a in nd.attribute}
        check_limits(name, nd.op_type, attrs, problems)

        nodes.append({
            "name": name,
            "op_type": nd.op_type,
            "inputs": list(nd.input),
            "outputs": list(nd.output),
            "attributes": attrs,
        })
    return nodes

def dumps(doc):
    """
    Serialize to JSON, keeping lists of numbers on a single line.

    Without this, one shape spans six lines and ResNet's graph.json grows to 2500 lines, defeating the
    reason JSON was chosen - that it's human readable.
    """
    text = json.dumps(doc, indent=2)
    text = re.sub(r"\[\s+([^\[\]{}]+?)\s+\]",
                  lambda m: "[" + " ".join(m.group(1).split()) + "]", text)
    return text + "\n"

def export(src, out_dir):
    """
    Export a .onnx into a graph.json + weights.bin pair and return the document.
    
    Everything is built in memory before anything is written, so a rejected model
    leaves behind no half-written artifact that looks like success.
    """
    model = onnx.load(str(src))
    graph = model.graph
    problems = []

    weights, initializers = pack_weights(graph, problems)
    init_names = {e["name"] for e in initializers}

    # Older ONNX IR lists initializers amog the graph inputs too. Filter them out.
    inputs = [value_info(vi, problems) for vi in graph.input if vi.name not in init_names]
    outputs = [value_info(vi, problems) for vi in graph.output]
    nodes = build_nodes(graph, problems)

    opset = next((o.version for o in model.opset_import if o.domain == ""), None)
    if opset is None:
        problems.append("model has no default ONNX opset domain")
    if problems:
        raise Rejected("\n".join(f"  - {p}" for p in problems))
    
    doc = {
        "schema_version": SCHEMA_VERSION,
        "model": {
            "name": src.stem,
            "source": str(src),
            "opset": opset,
            "ir_version": model.ir_version,
            "producer": model.producer_name,
        },
        "weights_file": "weights.bin",
        "weights_nbytes": len(weights),
        "inputs": inputs,
        "outputs": outputs,
        "initializers": initializers,
        "nodes": nodes,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "weights.bin").write_bytes(weights)
    (out_dir / "graph.json").write_text(dumps(doc))
    return doc

def main():
    """
    CLI: export one model, by default into artifacts/<model_name>/.
    """
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model", type=Path)
    ap.add_argument("-o", "--out", type=Path, default=None)
    args = ap.parse_args()

    out_dir = args.out or Path("artifacts") / args.model.stem
    try:
        doc = export(args.model, out_dir)
    except Rejected as e:
        print(f"export rejected for {args.model}:\n{e}", file=sys.stderr)
        raise SystemExit(1)

    print(f"{out_dir}/graph.json   {len(doc['nodes'])} nodes, "
            f"{len(doc['initializers'])} initializers")
    print(f"{out_dir}/weights.bin  {doc['weights_nbytes']} bytes")


if __name__ == "__main__":
    main()