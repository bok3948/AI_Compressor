import subprocess

import torch

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


  



            




