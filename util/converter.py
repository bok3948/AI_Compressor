import subprocess

import torch

from executorch.backends.xnnpack.partition.xnnpack_partitioner import (
    XnnpackPartitioner,
)
from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower
from torch.export import export, export_for_training, Dim

def torch_onnx_convert(model, model_name="convnext_small", dummy_size=(1, 3, 224, 224), save_dir="./" ):
    model.eval()
    x = torch.randn(dummy_size, requires_grad=True)

    file_name =  model_name + ".onnx"
    torch.onnx.export(model,              
                    x,                         
                    save_dir + "/"  + file_name,
                    export_params=True,        
                    opset_version=20,          
                    do_constant_folding=True,  
                    input_names = ['input'],   
                    output_names = ['output'], 
                    dynamic_axes={'input' : {0 : 'batch_size'},    
                                    'output' : {0 : 'batch_size'}},
                    )
    print(f"Model {model_name} is converted to ONNX format as {save_dir}/{file_name}")
    return save_dir + "/"  + file_name

def onnx_prepro(onnx_model_path):
    
    #preprocess the model. fusion folding etc
    command = [
        "python", "-m", "onnxruntime.quantization.preprocess",
        "--input", onnx_model_path,
        "--output", onnx_model_path
    ]

    result = subprocess.run(command, check=False)
    print(f"Model {onnx_model_path} is preprocessed")
    return onnx_model_path

def ExTorch_converter(model, dummy_size, file_name):

    edge = to_edge_transform_and_lower(
        model,
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
        partitioner=[XnnpackPartitioner()],
    )

    et_program= edge.to_executorch()

    # Save the ExecuTorch program to a file.
    with open(file_name, "wb") as file:
        et_program.write_to_file(file)



  



            




