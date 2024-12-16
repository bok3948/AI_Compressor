from compress.engine import evaluate 
from util.profiler import profiler
from onnx_inference import ONNX_inference
from prettytable import PrettyTable

def Benchmark(model_type, model, model_path, data_loader_val, dummy_size, device, args):
    #modell type: torch or onnx
    pf = profiler(dummy_size=dummy_size, model_type=model_type)
    if model_type == "torch":
        torch_stats = evaluate(data_loader_val, model, device, args)
        summary = pf.summary(model, model_path, device, args.model)
        summary.update(torch_stats)
    
    elif model_type == "onnx":
        from onnx_inference import ONNX_inference
        onnx_stats = ONNX_inference(data_loader_val, model_path, args)
        summary = pf.summary(None, model_path, device, args.model)
        summary.update(onnx_stats)

    elif model_type == "executorch":
        from extorch_inference import ExTorch_inference
        et_stats = ExTorch_inference(data_loader_val, model_path, args)
        summary = pf.summary(None, model_path, device, args.model)
        summary.update(et_stats)

    else:
        raise ValueError("Model type not supported")
    
    table = PrettyTable()

    table.field_names = list(summary.keys())
    table.add_row(list(summary.values()))
    print(table)
    
    return summary

    
    
