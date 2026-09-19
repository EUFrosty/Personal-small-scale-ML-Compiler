#!/usr/bin/env python3
"""Create a small conv net directly through onnx.helper and save it in models/."""

from pathlib import Path
import numpy as np
import onnx
from onnx import TensorProto, checker, helper, numpy_helper, shape_inference

OPSET = 17
IR_VERSION = 13 # onnxruntime 1.30 refuses everything above 13, and onnx 1.23 uses 14 by default if not specified, so we add this as a fix
SEED = 0
OUT = Path(__file__).resolve().parents[1] / "models" / "small_convnet.onnx"


def conv_params(rng, out_ch, in_ch, k):
    fan_in = in_ch * k * k
    scale = np.float32(np.sqrt(2.0 / fan_in))
    w = rng.standard_normal((out_ch, in_ch, k, k), dtype=np.float32) * scale
    b = rng.standard_normal(out_ch, dtype=np.float32) * np.float32(0.1)
    return w, b

def main():
    rng = np.random.default_rng(SEED)

    # 13 and 17 are primes, 30 is not divisible by 8/16/32 -> every tile size makes a tile
    N, C0, H, W = 1, 3, 30, 30
    C1, C2 = 13, 17

    w1, b1 = conv_params(rng, C1, C0, 3)
    w2, b2 = conv_params(rng, C1, C1, 3)
    w3, b3 = conv_params(rng, C2, C1, 3)

    initializers = [
        numpy_helper.from_array(w1, "conv1.w"),
        numpy_helper.from_array(b1, "conv1.b"),
        numpy_helper.from_array(w2, "conv2.w"),
        numpy_helper.from_array(b2, "conv2.b"),
        numpy_helper.from_array(w3, "conv3.w"),
        numpy_helper.from_array(b3, "conv3.b"),
    ]

    nodes = [
        helper.make_node("Conv", ["input", "conv1.w", "conv1.b"], ["c1"], name="conv1",
                         kernel_shape=[3, 3], pads=[1, 1, 1, 1], strides=[1, 1]),
        helper.make_node("Relu", ["c1"], ["h"], name="relu1"),

        helper.make_node("Conv", ["h", "conv2.w", "conv2.b"], ["c2"], name="conv2",
                         kernel_shape=[3, 3], pads=[1, 1, 1, 1], strides=[1, 1]),
        helper.make_node("Add", ["c2", "h"], ["a"], name="add1"),
        helper.make_node("Relu", ["a"], ["r"], name="relu2"),

        helper.make_node("Conv", ["r", "conv3.w", "conv3.b"], ["c3"], name="conv3",
                         kernel_shape=[3, 3], pads=[0, 0, 0, 0], strides=[1, 1]),
        helper.make_node("Relu", ["c3"], ["output"], name="relu3"),
    ]

    graph = helper.make_graph(
        nodes,
        "small_convnet",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [N, C0, H, W])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [N, C2, H - 2, W - 2])],
        initializer=initializers,
    )

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", OPSET)])
    model.ir_version = IR_VERSION
    model.producer_name = "mlc-build-small-model"
    model = shape_inference.infer_shapes(model)
    checker.check_model(model, full_check=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(OUT))

    print(f"upisano: {OUT}")
    for vi in model.graph.value_info:
        dims = [d.dim_value for d in vi.type.tensor_type.shape.dim]
        print(f"  {vi.name:>6s}  {dims}")

if __name__ == "__main__":
    main()