import onnx
import pytest
import torch
import norse.torch as snn


def test_export_onnx_li():
    p = snn.LIParameters(tau_syn_inv=torch.as_tensor(42))
    net = snn.SequentialState(snn.LICell(p))
    inp = torch.randn(2, 1, 10)
    torch.onnx.export(net, inp, "snn_li.onnx")

    loaded = onnx.load("snn_li.onnx")
    onnx.checker.check_model(loaded)


def test_export_onnx_lif():
    p = snn.LIFParameters(tau_syn_inv=torch.as_tensor(42), v_th=torch.as_tensor(0.6))
    net = snn.SequentialState(snn.LIFCell(p))
    inp = torch.randn(2, 1, 10)
    torch.onnx.export(net, inp, "snn_lif.onnx")


@pytest.mark.parametrize(
    "name,cell,params",
    [
        ("lif_ex", snn.LIFExCell, snn.LIFExParameters(tau_syn_inv=torch.as_tensor(42))),
        (
            "lif_adex",
            snn.LIFAdExCell,
            snn.LIFAdExParameters(tau_syn_inv=torch.as_tensor(42)),
        ),
        ("lsnn", snn.LSNNCell, snn.LSNNParameters(tau_syn_inv=torch.as_tensor(42))),
        (
            "lif_box",
            snn.LIFBoxCell,
            snn.LIFBoxParameters(tau_mem_inv=torch.as_tensor(42)),
        ),
    ],
)
def test_export_onnx_integer_params(name, cell, params):
    # integer parameter tensors must not abort the dynamo exporter (issue #447)
    net = snn.SequentialState(cell(params))
    inp = torch.randn(2, 1, 10)
    torch.onnx.export(net, inp, f"snn_{name}.onnx")

    loaded = onnx.load(f"snn_{name}.onnx")
    onnx.checker.check_model(loaded)


def test_export_onnx_integer_params_coba_lif():
    p = snn.CobaLIFParameters(tau_syn_exc_inv=torch.as_tensor(42))
    net = snn.SequentialState(snn.CobaLIFCell(10, 10, p))
    inp = torch.randn(2, 1, 10)
    torch.onnx.export(net, inp, "snn_coba_lif.onnx")

    loaded = onnx.load("snn_coba_lif.onnx")
    onnx.checker.check_model(loaded)
