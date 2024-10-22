import copy
import json

import torch

from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm import utils

from compress.engine import train_one_epoch, evaluate
from util import misc
from util.scheduler import prune_scheduler

from compress.structured_prune import St_Prune


class Retrain(object):
    def __init__(self, ori_model, data_loader_train, data_loader_val,criterion, device, output_dir, dummy_size, prune, args):
        self.ori_model = ori_model
        self.criterion = criterion
        self.output_dir = output_dir
        self.data_loader_train = data_loader_train
        self.data_loader_val = data_loader_val
        self.device = device
        self.dummy_size = dummy_size
        self.prune = prune
        self.args = args

        if self.args.do_KD:
            self.teacher = self.ori_model.to(self.device)
            self.teacher.eval()
        else:
            self.teacher = None

    def __call__(self, model, args):

        if self.prune:
            prune_scheduler_fn = prune_scheduler(args.total_iters, args.epochs)
        else:
            optimizer = create_optimizer(args, model)
            lr_scheduler, _ = create_scheduler(args, optimizer)

        loss_scaler = None
        iter = 0
        max_accuracy, best_model = 0, model
        for epoch in range(args.epochs):

            if  self.prune and prune_scheduler_fn(epoch):
                model = St_Prune(model, self.dummy_size, self.device, args)
                optimizer = create_optimizer(args, model)
                lr_scheduler, _ = create_scheduler(args, optimizer)
                iter += 1
                if iter == args.total_iters:
                    max_accuracy = 0

            train_stats = train_one_epoch(self.data_loader_train, model, 
                optimizer, self.criterion, None, self.device,
                epoch, None, self.teacher, args
            )

            if lr_scheduler is not None:
                lr_scheduler.step(epoch)

            val_stats = evaluate(self.data_loader_val, model, self.device, args)

            if max_accuracy < val_stats["acc1"]:
                max_accuracy = val_stats["acc1"]
                best_model = model

        return best_model, max_accuracy