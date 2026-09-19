#!/usr/bin/env python3
"""Runs a single ONNX operator through onnxruntime - for checking the calculations."""

import numpy as np
import onnxruntime as ort
from onnx import TensorProto, helper

OPSET = 17
IR_VERSION = 13

ort.set_default_logger_severity(3)


def run_op(op_type, inputs, **attrs):
    names = list(inputs)
    graph = helper.make_graph(
        [helper.make_node(op_type, names, ["Y"], **attrs)],
        "single",
        [helper.make_tensor_value_info(n, TensorProto.FLOAT, list(inputs[n].shape))
         for n in names],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, None)],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", OPSET)])
    model.ir_version = IR_VERSION
    sess = ort.InferenceSession(model.SerializeToString(),
                                providers=["CPUExecutionProvider"])
    return sess.run(["Y"], {n: inputs[n].astype(np.float32) for n in names})[0]


if __name__ == "__main__":
    X = np.arange(1, 17, dtype=np.float32).reshape(1, 1, 4, 4)
    W = np.ones((1, 1, 3, 3), dtype=np.float32)
    Y = run_op("MaxPool", {"X": X}, kernel_shape=[3,3], pads=[1,1,1,1], strides=[1,1])
    print(Y.shape)
    print(Y[0, 0])