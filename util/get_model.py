import os

import torch

import tensorflow as tf

from timm import create_model
from huggingface_hub import hf_hub_download
from transformers import AutoModel

from model import *

def get_torch_model(model_name=None, args=None):   
    if len(args.pretrained) > 0:
        path = args.pretrained

        # load model
        print(f"Creating  model: {model_name} from {path}")
        model = create_model(
            model_name,
            pretrained=True,
            num_classes=args.nb_classes,
        )

        if path.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                path, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(path, map_location='cpu')

        try:
            msg = model.load_state_dict(checkpoint['model'], strict=False)
            print(msg)
        except:
            try:
                msg = model.load_state_dict(checkpoint, strict=False)
                print(msg)
            except:
                model = checkpoint
                
        return model
        
    elif len(args.hf_hub_repo_id) > 0:

        file_path = hf_hub_download(repo_id=args.hf_hub_repo_id, filename=args.hf_hub_filename)
        
        _, file_extension = os.path.splitext(file_path)
        if file_extension == ".pt":
            print(f"Loading .pt model from {file_path}...")
            model = torch.load(file_path)  
        elif file_extension == ".bin":
            print(f"Loading .bin model from {file_path}...")
            model = AutoModel.from_pretrained(file_path) 
        else:
            raise ValueError("Unsupported file extension")
    
        return model
    else:
         # load model
        print(f"Creating timm pretrained model: {model_name}")
        model = create_model(
            model_name,
            pretrained=True,
            num_classes=args.nb_classes,
        )
        return model
