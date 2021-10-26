import sys
import argparse

from numpy.core.shape_base import block
sys.path.append("/pool0/wynwyn/dragon-project/src/dlrm")
sys.path.append("/pool0/wynwyn/dragon-project/src")

from mapper.mappings import illusion_mapping
from sweep.gen_testbench import gen_resnet, gen_bert
from math import ceil

N_GENERATION = 11
N_CHIP_0 = 2
N_CHIP_SCALE = 1.5

def calc_resnet_message(basic_size, blocktype, real_model):
    ''' Calculate resent message cost
    
    basic_size: model size for generation 0
    blocktype: "BasicBlock" or "Bottleneck"
    real_model: bool, whether construct a model in real scale size or use deeper/wider in illusio_mapping
    '''
    
    if real_model:
        for g in range(N_GENERATION):
            n_param, n_layer, resnet_graph = gen_resnet([i*(2**g) for i in basic_size], blocktype, groups=1, width_per_group=64)
            n_chip = int(N_CHIP_0 * (N_CHIP_SCALE ** g)) 
            capacity = ceil(n_param / n_chip) 
            print("{:>12.2e},".format(n_param), end=" ")
            illusion_mapping(resnet_graph, n_chip, 1, capacity, True, False)
    else:
        n_param, n_layer, resnet_graph = gen_resnet(basic_size, blocktype, groups=1, width_per_group=64)
        for g in range(N_GENERATION):
            n_chip = int(N_CHIP_0 * (N_CHIP_SCALE ** g))
            depth = 2**g
            capacity = ceil(n_param * depth / n_chip) 
            print("{:>12.2e},".format(n_param * depth), end=" ")
            illusion_mapping(resnet_graph, n_chip, depth, capacity, True, False)
        for g in range(N_GENERATION):
            n_chip = int(N_CHIP_0 * (N_CHIP_SCALE ** g))
            width = 2**g
            capacity = ceil(n_param * width / n_chip) 
            print("{:>12.2e},".format(n_param * width), end=" ")
            illusion_mapping(resnet_graph, n_chip, width, capacity, False, True)


    return

def calc_bert_message(basic_size, real_model):
    ''' Calculate Bert message cost
    
    basic_size: model size for generation 0
    real_model: bool, whether construct a model in real scale size or use deeper/wider in illusio_mapping
    '''
    VOCAB_SIZE = 30522 * 768
    if real_model:
        for g in range(N_GENERATION):
            n_param, n_layer, resnet_graph = gen_bert(basic_size*(2**g))
            n_chip = int(N_CHIP_0 * (N_CHIP_SCALE ** g)) 
            capacity = ceil((n_param - VOCAB_SIZE) / n_chip) 
            print("{:>12.2e},".format(n_param), end=" ")
            illusion_mapping(resnet_graph, n_chip, 1, capacity, True, False)
    else:
        n_param, n_layer, resnet_graph = gen_bert(basic_size)
        for g in range(N_GENERATION):
            n_chip = int(N_CHIP_0 * (N_CHIP_SCALE ** g))
            depth = 2**g
            capacity = ceil((n_param - VOCAB_SIZE) * depth / n_chip) 
            print("{:>12.2e},".format((n_param - VOCAB_SIZE) * depth + VOCAB_SIZE), end=" ")
            illusion_mapping(resnet_graph, n_chip, depth, capacity, True, False)


    return

if __name__ == "__main__":
    print("{:>12}, {:>12}, {:>12}, {:>12}, {:>12}, {:>12}".format("n_param", "message_cost", "avg_bound", "max_bound", "violate_avg_bound", "violate_max_bound"))

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, choices=["bert", "resnet"], help="model to be swept")
    parser.add_argument("-r", "--real_model", action="store_true", default=False, help="use real model")
    parser.add_argument("-b", "--blocktype", type=str, choices=["BasicBlock", "Bottleneck"], help="resnet block type")
    parser.add_argument("-s", "--chip_scale", type=float, help="#chip scaling factor")
    args = parser.parse_args()
    if args.model == "resnet":
        if args.blocktype == "BasicBlock":
            N_CHIP_SCALE = args.chip_scale
            calc_resnet_message([2, 2, 2, 2], "BasicBlock", args.real_model)
        if args.blocktype == "Bottleneck":
            N_CHIP_SCALE = args.chip_scale
            calc_resnet_message([3, 4, 6, 3], "Bottleneck", args.real_model)

    elif args.model == "bert":
        calc_bert_message(12, args.real_model)



