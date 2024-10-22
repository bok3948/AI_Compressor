import torch
import torch.nn.functional as F

import timm 

import torch_pruning as tp

import util.misc as misc


def St_Prune(model, dummy_size, device, args):
    imp = tp.importance.GroupNormImportance()
    print("Pruning %s..."%args.model)
        
    example_inputs = torch.randn(*dummy_size).to(device)

    layer_names = misc.get_layer(model, [])
    ignored_layers = [misc.get_module_by_name(model, layer_names[-1])]

    if "deit" in args.model:
        num_heads = get_head_data(model)

    pruner = tp.pruner.MetaPruner(
                    model, 
                    example_inputs, 
                    global_pruning=args.global_pruning, # If False, a uniform pruning ratio will be assigned to different layers.
                    importance=imp, # importance criterion for parameter selection
                    iterative_steps=1, # the number of iterations to achieve target pruning ratio
                    pruning_ratio=float(args.pruning_ratio/args.total_iters) , # target pruning ratio
                    pruning_ratio_dict={},
                    ignored_layers=ignored_layers,
                    
                    num_heads=num_heads if "deit" in args.model else {},
                    prune_num_heads=True if "deit" in args.model else False,
                    prune_head_dims=False,
                    head_pruning_ratio=float(1/6) if "deit" in args.model else 0.0,
                    unwrapped_parameters=[(model.cls_token, 2), (model.pos_embed, 2)] if "deit" in args.model else [],
                )
    for g in pruner.step(interactive=True):
        g.prune()

    if "deit" in args.model:
        change_forward(model)

    return model


def change_forward(module, parent_name=""):
    for name, child in module.named_children():
        layer_full_name = f"{parent_name}.{name}" if parent_name else name
        
        if isinstance(child, timm.models.vision_transformer.Attention):
            qkv_name = f"{layer_full_name}.qkv"
            #change forward function of Attention module
            child.forward = forward.__get__(child, timm.models.vision_transformer.Attention) # https://stackoverflow.com/questions/50599045/python-replacing-a-function-within-a-class-of-a-module
            child.num_heads = int(child.qkv.out_features // (3 * child.head_dim))

        change_forward(child, layer_full_name)

def get_head_data(module, parent_name="", head_data={}):

    for name, child in module.named_children():
        layer_full_name = f"{parent_name}.{name}" if parent_name else name
        
        if isinstance(child, timm.models.vision_transformer.Attention):
            head_data.update({child.qkv: child.num_heads})
        get_head_data(child, layer_full_name, head_data)

    return head_data

def forward(self, x):
    """https://github.com/huggingface/pytorch-image-models/blob/054c763fcaa7d241564439ae05fbe919ed85e614/timm/models/vision_transformer.py#L79"""
    B, N, C = x.shape

    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    q, k = self.q_norm(q), self.k_norm(k)

    if self.fused_attn:
        x = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_drop.p,
        )
    else:
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = attn @ v

    x = x.transpose(1, 2).reshape(B, N, -1) # original implementation: x = x.transpose(1, 2).reshape(B, N, C)
    x = self.proj(x)
    x = self.proj_drop(x)
    return x



