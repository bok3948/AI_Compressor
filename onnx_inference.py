import torch

import onnxruntime 

from timm.utils import accuracy

import util.misc as misc

#Hardware acceleration
execution_provider = {
    "default_cpu": ["CPUExecutionProvider"],
    "nvidia_gpu": ["TensorrtExecutionProvider", "CUDAExecutionProvider"],
    "intel_cpu": ["OpenVINOExecutionProvider", "DnnlExecutionProvider", "CPUExecutionProvider"],
    # "android_cpu": ["NNAPIExecutionProvider", "CPUExecutionProvider"],
    # "apple_cpu": ["CoreMLExecutionProvider", "CPUExecutionProvider"],
}

def ONNX_inference(onnx_loader_val, onnx_path, args):
    EP_list = execution_provider[args.device_type]
    ort_session = onnxruntime.InferenceSession(onnx_path, providers=EP_list)
    metric_logger = misc.MetricLogger(delimiter="\t")
    header = 'ONNX inference:'

    for inputs, labels in metric_logger.log_every(onnx_loader_val, 100, header):

        inputs_np = inputs.numpy()

        ort_inputs = {ort_session.get_inputs()[0].name: inputs_np}
        ort_outs = ort_session.run(None, ort_inputs)

        logits_np = ort_outs[0]
        logits_tensor = torch.from_numpy(logits_np)
        logits_tensor = torch.nn.functional.softmax(logits_tensor, dim=1)

        acc1, acc5 = accuracy(logits_tensor, labels, topk=(1, 5))

        batch_size = inputs.shape[0]
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}