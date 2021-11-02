from torchvision.models import resnet
from torchvision.models.resnet import ResNet, BasicBlock, Bottleneck
from transformers import BertModel, BertConfig
import torch
import sys
import numpy as np

sys.path.append("/pool0/wynwyn/dragon-project/src")
from ir.trace import trace

def gen_resnet(layers, block, **kwargs):
    """Generate Resnet models

    layers: list, len(layer) = 4
    block: BasicBlock or Bottleneck
    groups: int 
    width_per_group: int, 64*k
    
    For BasicBlock, resnet{sum(layer)*2+2}
    For Bottleneck, resnet{sum(layer)*3+2}_{group}x{width_per_group}d
    """
    if block == "BasicBlock":
        model = ResNet(BasicBlock, layers)
        n_layer = sum(layers)*2 + 2
    elif block == "Bottleneck":
        model = ResNet(Bottleneck, layers, groups=kwargs["groups"], width_per_group=kwargs["width_per_group"])
        n_layer = sum(layers)*3 + 2
    else:
        raise(ValueError, "block should be BasicBlock or Bottleneck")
    n_param = 0
    for param in model.parameters():
        n_param += np.prod(param.size())
    inputs = torch.randn([1, 3, 224, 224])
    resnet_graph = trace(model.eval(), inputs)
    # return resnet_graph, n_param
    return n_param, n_layer, resnet_graph

def gen_bert(layers):
    configuration = BertConfig(num_hidden_layers=layers, num_attention_heads=1)
    model = BertModel(configuration)

    indexed_tokens = [
        101,
        2627,
        1108,
        3104,
        1124,
        15703,
        136,
        102,
        3104,
        1124,
        15703,
        1108,
        170,
        16797,
        8284,
        102,
    ]

    tokens_tensor = torch.tensor([indexed_tokens])
    model(tokens_tensor)

    bert_graph = trace(model.eval(), tokens_tensor)
    n_param = 0
    for param in model.parameters():
        n_param += np.prod(param.size())
    return n_param, len(list(model.parameters())), bert_graph

