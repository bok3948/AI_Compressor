"""
profile the model 
"""
import os
import time
import numpy as np

import torch

import onnxruntime

class profiler(object):
    def __init__(self, dummy_size=(1, 3, 224, 224), model_type="torch"):   
        self.dummy_size = dummy_size
        self.model_type = model_type

    def torch_model_latency(self, model):
        device = "cpu"
        dummy_size = self.dummy_size

        dummy_input = torch.randn(*self.dummy_size, dtype=torch.float).to(device)
        model = model.to(device)
        repetitions = 100
        warmup = 50
        timings = np.zeros((repetitions, 1))

        with torch.no_grad():
            for _ in range(warmup):
                _ = model(dummy_input)

            for rep in range(repetitions):
                start = time.time()
                _ = model(dummy_input)
                end = time.time()
                timings[rep] = (end - start) * 1000  # Convert to milliseconds

        mean_time = np.mean(timings)
        name = "cpu_lantency" + "@bs" + str(1) + "_ms"
        return round(mean_time, 4)
    
    def onnx_model_latency(self, model_path):
        session = onnxruntime.InferenceSession(model_path)
        input_name = session.get_inputs()[0].name

        total = 0.0
        runs = 100
        input_data = np.ones(self.dummy_size, np.float32)
        # Warming up
        for _ in range(50):
            _ = session.run([], {input_name: input_data})
        for i in range(runs):
            start = time.time()
            _ = session.run([], {input_name: input_data})
            end = (time.time() - start) * 1000
            total += end
        total /= runs
        return round(total, 4)
    
    def model_size(self, model_path):
        size = os.path.getsize(model_path) / 1e6  # 파일 크기를 바이트에서 MB로 변환
        return size
    
    def torch_model_size(self, mdl):
        torch.save(mdl.state_dict(), "tmp.pt")
        size = os.path.getsize("tmp.pt")/1e6
        os.remove('tmp.pt')
        return round(size, 2)
    
    def summary(self, model, model_path, device, name):
        summary_dict = {}
        
        if self.model_type == "torch":
            model = model.to(device)
            summary_dict["name"] = f"torch_{name}"
            summary_dict["latency(ms)"] = self.torch_model_latency(model)
            summary_dict["size(mb)"] = self.torch_model_size(model)

        elif self.model_type == "onnx":
            summary_dict["name"] = f"onnx_{name}"
            summary_dict["latency(ms)"] = self.onnx_model_latency(model_path)
            summary_dict["size(mb)"] = self.model_size(model_path)

        else:
            raise ValueError("model_type should be 'torch' or 'onnx'")
        
        return summary_dict
    



