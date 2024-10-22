from functools import partial

import torch

from torch.ao.quantization import get_default_qat_qconfig_mapping, get_default_qat_qconfig
from torch.quantization.quantize_fx import prepare_fx, convert_fx, prepare_qat_fx
from torch.ao.quantization import QConfigMapping, QConfig, FakeQuantize, MovingAverageMinMaxObserver, MovingAveragePerChannelMinMaxObserver

from compress.retrain import Retrain
from compress.engine import evaluate

# Pytorch fx mode 
def QAT(model, ori_model, data_loader_train, data_loader_val, criterion, device, dummy_size, args):
    model.eval()

    #qconfig_mapping to apply quantization only to specific layers (e.g., linear and conv layers)  quantized only the linear and conv layers for compamtiability with the onnx
    if "deit" in args.model: 
        qconfig_mapping = QConfigMapping().set_object_type(torch.nn.Linear, get_default_qat_qconfig("fbgemm")).set_object_type(torch.nn.Conv2d, get_default_qat_qconfig("fbgemm"))
    else:
        qconfig_mapping = QConfigMapping().set_global(get_default_qat_qconfig("fbgemm"))
    #prepare
    dummy_input = torch.randn(*dummy_size, dtype=torch.float).to("cpu")
    model_prepared = prepare_qat_fx(model.to("cpu"), qconfig_mapping, dummy_input)

    #PTQ
    print(f"-"*20 + "PTQ" + "-"*20)
    evaluate(data_loader_train, model_prepared.to(device), device, args)

    #train 
    print(f"-"*20 + "QAT" + "-"*20)
    
    args.lr = args.qat_lr
    args.epochs = args.qat_epochs

    criterion = torch.nn.CrossEntropyLoss().to(args.device)
    retrain_fn = Retrain(ori_model, data_loader_train, data_loader_val, criterion, device, args.output_dir, dummy_size, False, args)
    model_trained, _ = retrain_fn(model_prepared.to(device), args)

    # Convert to quantized model
    model_trained = model_trained.to("cpu")
    model_quantized = convert_fx(model_trained)
    model_quantized.eval()
  
    return model_quantized
