import torch

from util import misc
from timm.utils import accuracy

from executorch.runtime import Runtime

def ExTorch_inference(data_loader, file_path, args):
    runtime = Runtime.get()
    program = runtime.load_program(file_path)
    method = program.load_method("forward")

    metric_logger = misc.MetricLogger(delimiter="\t")
    header = 'Test:'

    for inputs, labels in metric_logger.log_every(data_loader, int(len(data_loader)//2), header):
        inputs, labels = inputs.to("cpu", non_blocking=True), labels.to("cpu", non_blocking=True)

        logits = method.execute([inputs])[0]
        logits = torch.nn.functional.softmax(logits, dim=1)
        acc1, acc5 = accuracy(logits, labels, topk=(1, 5))

        batch_size = inputs.shape[0]
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

    metric_logger.synchronize_between_processes()
    
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f}'
        .format(top1=metric_logger.acc1, top5=metric_logger.acc5))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    
