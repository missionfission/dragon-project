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
    input_size: list, len(input_size) = 4
    
    For BasicBlock, resnet{sum(layer)*2+2}
    For Bottleneck, resnet{sum(layer)*3+2}_{group}x{width_per_group}d
    """
    if block == "BasicBlock":
        model = ResNet(BasicBlock, layers)
        n_layer = sum(layers)*2 + 2
        print(n_layer, layers)
    elif block == "Bottleneck":
        model = ResNet(Bottleneck, layers, groups=kwargs["groups"], width_per_group=kwargs["width_per_group"])
        n_layer = sum(layers)*3 + 2
    else:
        raise(ValueError, "block should be BasicBlock or Bottleneck")
    n_param = 0
    for param in model.parameters():
        n_param += np.prod(param.size())
    inputs = torch.randn(kwargs["input_size"])
    resnet_graph = trace(model.eval(), inputs)
    # return resnet_graph, n_param
    return n_param, n_layer


def sweep():
    groups = [1, 16] #[1, 2, 4, 8, 16]
    width_per_group = [64, 64*(2**3)] #[64*(2**i) for i in range(11)]
    input_size = [1, 3, 224, 224]
    layer_sizes = [[1, 1, 1, 1], [1024, 1024, 1024, 1024]]#[(2**i, 2**j, 2**k, 2**l) for i in range(11) for j in range(11) for k in range(11) for l in range(11)]

    model_list = []

    for l1, l2, l3, l4 in layer_sizes:
        print(l1, l2, l3, l4)
        # BasicBlock
        n_param, n_layer = gen_resnet([l1, l2, l3, l4], "BasicBlock", input_size=input_size)
        model_list.append([n_param, n_layer, "BasicBlock_{}_{}_{}_{}".format(l1, l2, l3, l4)])
        # Bottleneck
        for group in groups:
            for w in width_per_group:
                n_param, n_layer = gen_resnet([l1, l2, l3, l4], "Bottleneck", input_size=input_size, groups=group, width_per_group=w)
                model_list.append([n_param, n_layer, "Bottleneck_{}_{}_{}_{}_{}g_{}b".format(l1, l2, l3, l4, group, w)])
    return model_list

def gen_bert():
    configuration = BertConfig(num_hidden_layers=12)
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
    print(model.eval())
    print(len(list(model.parameters())))


    model(tokens_tensor)

    bert_graph = trace(model, tokens_tensor)
    print(bert_graph)
    n_param = 0
    for param in model.parameters():
        n_param += np.prod(param.size())
    return n_param

if __name__ == "__main__":
    # 
    # print("{:e}".format(gen_bert()))
    # exit()
    model_list = [gen_resnet([3, 8, 36, 3], "Bottleneck", input_size=[1, 3, 224, 224], groups=1, width_per_group=64)]
    model_list = sorted(model_list, key=lambda x: x[0])
    print("#params min {:e}, max {:e}".format(model_list[0][0], model_list[-1][0]))
    sorted(model_list, key=lambda x: x[1])
    print("#layers min {}, max {}".format(model_list[0][1], model_list[-1][1]))


    # Chip size = n_param / n_chip * constant_scale
    # illusion_mapping(graph, n_chips, 1, n_param/n_chips, deeper=True, wider=False)
