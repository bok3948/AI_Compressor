# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import os
import copy
import json
import numpy as np

import torch
from torchvision import datasets, transforms
from torchvision.datasets.folder import ImageFolder, default_loader

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


NORMALIZE_DICT = {
    'cifar10':  dict( mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010) ),
    'cifar100': dict( mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761) ),
    'cifar10_224':  dict( mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010) ),
    'cifar100_224': dict( mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761) ),
}


class INatDataset(ImageFolder):
    def __init__(self, root, train=True, year=2018, transform=None, target_transform=None,
                 category='name', loader=default_loader):
        self.transform = transform
        self.loader = loader
        self.target_transform = target_transform
        self.year = year
        # assert category in ['kingdom','phylum','class','order','supercategory','family','genus','name']
        path_json = os.path.join(root, f'{"train" if train else "val"}{year}.json')
        with open(path_json) as json_file:
            data = json.load(json_file)

        with open(os.path.join(root, 'categories.json')) as json_file:
            data_catg = json.load(json_file)

        path_json_for_targeter = os.path.join(root, f"train{year}.json")

        with open(path_json_for_targeter) as json_file:
            data_for_targeter = json.load(json_file)

        targeter = {}
        indexer = 0
        for elem in data_for_targeter['annotations']:
            king = []
            king.append(data_catg[int(elem['category_id'])][category])
            if king[0] not in targeter.keys():
                targeter[king[0]] = indexer
                indexer += 1
        self.nb_classes = len(targeter)

        self.samples = []
        for elem in data['images']:
            cut = elem['file_name'].split('/')
            target_current = int(cut[2])
            path_current = os.path.join(root, cut[0], cut[2], cut[3])

            categors = data_catg[target_current]
            target_current_true = targeter[categors[category]]
            self.samples.append((path_current, target_current_true))

    # __getitem__ and __len__ inherited from ImageFolder

def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    if args.data_set == 'CIFAR':
        args.input_size = 32
        dataset = datasets.CIFAR100(args.data_path, train=is_train, transform=transform, download=True)
        nb_classes = 100

    elif args.data_set == 'CIFAR10':
        args.input_size = 32
        # transform = cifar10_build_transform(is_train, args)
        transform = build_transform(is_train, args)
        dataset = datasets.CIFAR10(args.data_path, train=is_train, transform=transform, download=True)
        nb_classes = 10

    elif args.data_set == 'CIFAR10_224':
        args.input_size = 224
        # transform = cifar10_build_transform(is_train, args)
        transform = build_transform(is_train, args)
        dataset = datasets.CIFAR10(args.data_path, train=is_train, transform=transform, download=True)
        nb_classes = 10

    elif args.data_set == 'IMNET':
        args.input_size = 224
        root = os.path.join(args.data_path, 'train' if is_train else 'val')
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 1000
    elif args.data_set == 'INAT':
        dataset = INatDataset(args.data_path, train=is_train, year=2018,
                              category=args.inat_category, transform=transform)
        nb_classes = dataset.nb_classes

    elif args.data_set == 'INAT19':
        dataset = INatDataset(args.data_path, train=is_train, year=2019,
                              category=args.inat_category, transform=transform)
        nb_classes = dataset.nb_classes

    return dataset, nb_classes


def build_transform(is_train, args):
    resize_im = args.input_size > 32
    t = []
    if resize_im:
        size = int(args.input_size / args.eval_crop_ratio)
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)

def cifar10_build_transform(is_train, args):
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = transforms.Compose([
            transforms.RandomCrop(args.input_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(**NORMALIZE_DICT['cifar10'])
        ])

        return transform

    t = []
    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(**NORMALIZE_DICT['cifar10']))
    return transforms.Compose(t)


def build_calib_loader(dataset=None, num_samples=1024, seed=0, args=None):

    np.random.seed(seed)
    inds=np.random.permutation(len(dataset))[:num_samples]
    calib_set=torch.utils.data.Subset(copy.deepcopy(dataset), inds)
    # calib_set.dataset.transform = build_transform(is_train=False, args=args)

    return torch.utils.data.DataLoader(calib_set, batch_size=args.calib_batch_size, shuffle=False)


def build_data_loader(is_train=True, dataset=None, batch_size=10, args=None):

    if is_train:
        train_sampler = torch.utils.data.RandomSampler(dataset)
        
        data_loader_train = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, 
            num_workers=3,
            drop_last=True, sampler=train_sampler
        )
        return data_loader_train
    else:
        sampler_val = torch.utils.data.SequentialSampler(dataset)
        
        data_loader_val = torch.utils.data.DataLoader(
            dataset, sampler=sampler_val,
            batch_size=batch_size, 
            num_workers=3,
            drop_last=False
        )
        return data_loader_val

