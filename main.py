import json
import argparse
import copy
import numpy as np

import torch

from util.datasets import build_dataset, build_calib_loader, build_data_loader
from util import get_model
from util.converter import torch_onnx_convert

from compress.structured_prune import St_Prune

from compress.retrain import Retrain

from compress.onnx_ptq import ONNX_PTQ
from compress.qat import QAT
from benchmark import Benchmark
def get_args_parser():
    parser = argparse.ArgumentParser(description='AI Compression', add_help=False)

    parser.add_argument('--mode', default=1, choices=[0, 1], type=int, help='mode 0: post-training setting(pruning + PTQ) (NO retrain, No train data, with calibration dataset), mode 1: retraining setting(pruning + retraining + QAT) (retraining, with train dataset)')

    #setting
    parser.add_argument('--device_type', default='nvidia_gpu', choices=['default_cpu', 'nvidia_gpu', 'intel_cpu'], help='hardware acceleration for onnxruntime')
    parser.add_argument('--device', default='cuda', help='cpu vs cuda')

    #data load
    #/mnt/d/data/image/ILSVRC/Data/CLS-LOC
    parser.add_argument('--data-set', default='CIFAR10', choices=['CIFAR', 'IMNET', 'INAT', 'INAT19', "CIFAR10_224", "CIFAR10"])
    parser.add_argument('--data_path', default='/home/kimtaeho', type=str, help='path to ImageNet data')
    parser.add_argument('--eval-crop-ratio', default=0.875, type=float, help="Crop ratio for evaluation")
    parser.add_argument('--input-size', default=32, type=int, help='images input size')
    parser.add_argument('--batch_size', default=256, type=int, help='batch size for training')

    #calibration
    parser.add_argument('--num_samples', default=1000, type=int, help='size of the calibration dataset')
    parser.add_argument('--calib_batch_size', default=10, type=int, help='number of iterations for calibration')

    #model
    parser.add_argument('--model', default="resnet18", type=str, help='model name')
    parser.add_argument('--pretrained', default='', help='get pretrained weights from own checkpoint')
    parser.add_argument('--hf_hub_repo_id', default='', help='get pretrained weights from Hugging Face Hub')
    parser.add_argument('--hf_hub_file_name', default='', help='get pretrained weights from Hugging Face Hub')

    #prune 
    parser.add_argument('--total_iters', default=3, type=int, help='channel pruning ratio')
    parser.add_argument('--pruning_ratio', default=0.1, type=float, help='channel pruning ratio')
    parser.add_argument('--global_pruning', default=True, action='store_true', help='global pruning')

    #retrain for prune setting

    #optimizer
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--weight_decay', default=5e-4, type=float)

    # Learning rate schedule parameters
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--warmup-epochs', type=int, default=10, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--min-lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    
    #retrain for QAT setting
    parser.add_argument('--qat_lr', default=1e-5, type=float)
    
    #run
    parser.add_argument('--epochs', default=1, type=int)
    parser.add_argument('--qat_epochs', default=1, type=int)

    #distillation
    parser.add_argument('--do_KD', action='store_true', default=False,
                    help='do distillation')
    parser.add_argument('--KD_loss_weight', type=float, default=1.0)
    parser.add_argument('--temperature', type=float, default=2.0)

    #save and log
    parser.add_argument('--print_freq', default=500, type=int)
    parser.add_argument('--output_dir', default='./output_dir', type=str, help='path where to save scale, empty for no saving')
    
    return parser

def main(args):
    device = torch.device(args.device) 

    seed = 0
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    #get training dataset
    dataset_train, args.nb_classes = build_dataset(is_train=True, args=args)
    data_loader_train = build_data_loader(is_train=True, dataset=dataset_train, args=args)

    #get calibraition dataloader
    calib_loader = build_calib_loader(dataset_train, num_samples=args.num_samples, seed=seed, args=args)

    #validation dataset 
    dataset_val, _ = build_dataset(is_train=False, args=args)
    data_loader_val = build_data_loader(is_train=False, dataset=dataset_val, args=args)

    model = get_model.get_torch_model(args.model, args)
    model = model.to(device)

    ori_model = copy.deepcopy(model)
    ori_model = ori_model.to(device)
    save_args = vars(args)

    #pruning
    dummy_size = (1, 3, args.input_size, args.input_size)
    if args.mode == 0:
        #Prune
        print(f"="*20 + "Pruning" + "="*20)
        pruned_model = St_Prune(model, dummy_size, device, args)
        print(f"-"*20 + "Benchmarking pruned model" + "-"*20)
        pruned_stats = Benchmark("torch", pruned_model, None, data_loader_val, dummy_size, device, args)
        torch.save(pruned_model, f"./{args.model}_pruned.pth")

        #Quantization
        print(f"="*20 + "Quantization(PTQ)" + "="*20)
        onnx_path = torch_onnx_convert(pruned_model.to("cpu"), args.model, dummy_size)
        compress_path = ONNX_PTQ(onnx_path, calib_loader)

    elif args.mode == 1:
        #Prune
        print(f"="*20 + "Pruning + Retraining" + "="*20)
        criterion = torch.nn.CrossEntropyLoss().to(args.device)
        retrain_fn = Retrain(ori_model, data_loader_train, data_loader_val, criterion, device, args.output_dir, dummy_size,  True, args)
        pruned_model, best_acc = retrain_fn(model, args)

        print(f"-"*20 + "Benchmarking pruned model" + "-"*20)
        pruned_stats = Benchmark("torch", pruned_model, None, data_loader_val, dummy_size, device, args)
        torch.save(pruned_model, f"./{args.model}_pruned.pth")

        #Quantization
        print(f"="*20 + "Quantization" + "="*20)
        compress_model = QAT(pruned_model, ori_model, data_loader_train, data_loader_val, criterion, device, dummy_size, args)
        compress_path = torch_onnx_convert(compress_model.to("cpu"), args.model, dummy_size)
    
    # Benchmark
    print(f"-"*20 + "Benchmarking Original model" + "-"*20)
    ori_stats = Benchmark("torch", ori_model, None, data_loader_val, dummy_size, device, args)

    print(f"-"*20 + "Benchmarking Compressed model" + "-"*20)
    com_stats = Benchmark("onnx", None, compress_path, data_loader_val, dummy_size, device, args)
    
    results= {
        "benchmark_setting": {"device": args.device, "input_size": dummy_size},
        "ori_model": ori_stats,
        "pruned_model": pruned_stats,
        "compressed_model": com_stats,
        "args": save_args
    }

    json.dump(results, open("./results.json", "w"), indent=4)


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)
