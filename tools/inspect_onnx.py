#!/usr/bin/env python3
"""Prints the op histograms and checks v1 constraints for .onnx file."""

import sys
from collections import Counter

import onnx
from onnx import TensorProto, numpy_helper, shape_inference

FLOAT = TensorProto.FLOAT

def dims_of(vi):
    return [
        d.dim_value if d.HasField("dim_value") else (d.dim_param or "?")
        for d in vi.type.tensor_type.shape.dim
    ]

def dtype_of(vi):
    return TensorProto.DataType.Name(vi.type.tensor_type.elem_type)

def main(path):
    model = shape_inference.infer_shapes(onnx.load(path))
    g = model.graph
    init_names = {i.name for i in g.initializer}

    print(f"file            {path}")
    print(f"ir_version      {model.ir_version}")
    for oi in model.opset_import:
        print(f"opset       domain={oi.domain or '(default)'} version={oi.version}")
    print(f"producer    {model.producer_name or '(none)'}")
    print(f"nodes       {len(g.node)}")

    print("\nop histogram")
    for op, n in sorted(Counter(nd.op_type for nd in g.node).items(),
                        key=lambda kv: (-kv[1], kv[0])):
        print(f"  {n:4d}  {op}")

    for label, seq in (("inputs", g.input), ("outputs", g.output)):
        print(f"\n{label}")
        for vi in seq:
            if vi.name not in init_names:
                print(f"  {vi.name:22s} {dtype_of(vi):8s} {dims_of(vi)}")

    nbytes = sum(numpy_helper.to_array(i).nbytes for i in g.initializer)
    print(f"\ninitializers  {len(g.initializer)}  ({nbytes / 1e6:.1f} MB)")

    problems = []
    for vi in list(g.input) + list(g.output) + list(g.value_info):
        if vi.name in init_names:
            continue
        if any(not isinstance(d, int) for d in dims_of(vi)):
            problems.append(f"dynamic dim na '{vi.name}': {dims_of(vi)}")
        if vi.type.tensor_type.elem_type != FLOAT:
            problems.append(f"dtype {dtype_of(vi)} na '{vi.name}'")
    for i in g.initializer:
        if i.data_type != FLOAT:
            name = TensorProto.DataType.Name(i.data_type)
            problems.append(f"dtype {name} na initializeru '{i.name}'")

    print("\nv1 ogranicenja")
    for p in problems:
        print(f"  FAIL  {p}")
    if not problems:
        print("  OK    sve float32, svi shape-ovi statični")


if __name__ == "__main__":
    main(sys.argv[1])