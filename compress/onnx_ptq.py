import onnx
from onnxruntime.quantization import QuantType, quantize_static, calibrate, CalibrationDataReader, QuantFormat
from util.onnx_loader import ONNX_Loader
from util.converter import onnx_prepro

def ONNX_PTQ(model_path, calib_loader):

    onnx_calib_loader = ONNX_Loader(calib_loader)
    model_path = onnx_prepro(model_path) # preprocess the model batch folding layer fusion
    model = onnx.load(model_path)
    graph = model.graph

    nodes_to_quantize = []
    for node in graph.node:
        if node.op_type in ["Conv", "Gemm"]: 
            nodes_to_quantize.append(node.name)
    # print(f"node to quantize {nodes_to_quantize}")
    output_path = model_path.replace('.onnx', '_compress.onnx')
    quantize_static(
        model_input=model_path,
        model_output=output_path,
        per_channel=True,
        reduce_range=True,
        # nodes_to_quantize=nodes_to_quantize,
        nodes_to_quantize=nodes_to_quantize,
        nodes_to_exclude=None,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        extra_options={"ActivationSymmetric": True,
                        "WeightSymmetric": True},
        calibration_data_reader=onnx_calib_loader,
        calibrate_method=calibrate.CalibrationMethod.MinMax,
        )
    print(f"Model quantized: {output_path}")
    return output_path
