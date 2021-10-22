'''
Count message size of FC layers
'''

import random
from math import ceil
from collections import namedtuple
import matplotlib.pyplot as plt
import sys
import kahypar as kahypar

RELAXED_RATIO = 1.01
MAX_N_CHIP = 9
N_MODEL = 20
# ACT_SIZE_RANGE = int(sys.argv[1]) # 2**13
ACT_SIZE_RANGE = 26
# N_LAYER_RANGE = int(sys.argv[2]) # 2**7
N_LAYER_RANGE = 100 # 2**7
# CONFIG = sys.argv[3] #/pool0/wynwyn/kahypar/config/km1_kKaHyPar_sea20.ini km1_rKaHyPar_sea20.ini km1_kKaHyPar-E_sea20.ini
CONFIG = "/pool0/wynwyn/kahypar/config/km1_kKaHyPar_sea20.ini" # km1_rKaHyPar_sea20.ini km1_kKaHyPar-E_sea20.ini

# DNN Illusion Heuristic Partition

def cnt_partition_msg(activations, n_chip, mem_size):
    '''
    Assume homogeneous chips
    - activations: list(), activation X_0 ~ X_N
    - n_chip: int, number of chips (patitions)
    - mem_size: int on-chip memory size
    '''

    cur_chip_id = 0
    cur_mem = mem_size
    cnt_message = 0

    for i, in_act in enumerate(activations[:-1]):
        
        if cur_mem == 0:
            if cur_chip_id + 1 == n_chip: 
                # Not Enought memory, either because of not enough total memory or corner partition cases
                return None
            cur_chip_id += 1
            cur_mem = mem_size

        out_act = activations[i+1]
        mem_req = in_act * out_act

        if mem_req <= cur_mem:
            if cur_mem == mem_size:
                # new chip -> calculate input message count
                if  cur_chip_id != 0: cnt_message += in_act # corner case: 1st chip
            cur_mem -= mem_req
        else:
            par_in_input = in_act >= out_act
            if par_in_input:
                fix_act = out_act
                par_act = in_act
            else:
                fix_act = in_act
                par_act = out_act
            
            while fix_act*par_act > cur_mem:
                if cur_mem >= fix_act:
                    num_sub_act = cur_mem // fix_act
                    par_act -= num_sub_act
                    cur_mem -= num_sub_act * fix_act
                    cnt_message += (num_sub_act + fix_act)
                    if cur_chip_id == 0: # corner case: 1st chip
                        if par_in_input:
                            cnt_message -= num_sub_act
                        else:
                            cnt_message -= fix_act
                if cur_chip_id + 1 == n_chip: 
                    # Not Enought memory, either because of not enough total memory or corner partition cases
                    return None
                cur_chip_id += 1
                cur_mem = mem_size
                
            cur_mem -= par_act * fix_act
            if par_in_input:
                cnt_message += par_act
            else:
                cnt_message += fix_act
    
    return cnt_message

def kahypar_partition_msg(activations, n_chip):

    cum_edge_indices = [0]
    edges = []
    offset = 0
    
    for i, act in enumerate(activations):
        if i == 0 or i == len(activations) - 1:
            if i == 0:
                for j in range(act):
                    edges += [activations[1] * k + j for k in range(activations[1])]
                    cum_edge_indices.append(cum_edge_indices[-1] + activations[1])
            if i == len(activations) - 1:
                for j in range(act):
                    edges += [offset + activations[-2] * j + k for k in range(activations[-2])]
                    cum_edge_indices.append(cum_edge_indices[-1] + activations[-2])
                offset += activations[-2] * activations[-1]
            continue

        pre_act = activations[i-1]
        post_act = activations[i+1]
        n_offset = offset + pre_act * act


        for j in range(act):
            edges += [offset + pre_act * j + k for k in range(pre_act)] + [n_offset + post_act * k + j for k in range(post_act)]
            cum_edge_indices.append(cum_edge_indices[-1] + pre_act + post_act)

        offset = n_offset

    num_nodes = offset
    num_edges = len(cum_edge_indices) - 1
    node_weights = [1 for _ in range(num_nodes)]
    edge_weights = [1 for _ in range(num_edges)]

    hypergraph = kahypar.Hypergraph(num_nodes, num_edges, cum_edge_indices, edges, n_chip, edge_weights, node_weights)

    context = kahypar.Context()
    context.loadINIconfiguration(CONFIG)

    context.setK(n_chip)
    context.setSeed(4)
    context.setEpsilon(RELAXED_RATIO - 1)

    # kahypar.partition(hypergraph, context)
    return

def gen_model(act_size_range=ACT_SIZE_RANGE, n_layer_range=N_LAYER_RANGE):
    activations = [random.randint(1, act_size_range) for _ in range(random.randint(4, n_layer_range))]
    return activations

def get_tot_mem(activations):
    return sum([activations[i] * activations[i+1] for i in range(len(activations)-1)])

def get_avg_act_size(activations):
    return ceil(sum(activations) / (len(activations) - 1))

def get_mem_size(activations, n_chip, relaxed_ratio):
    return ceil(get_tot_mem(activations) * relaxed_ratio / n_chip)



Setting = namedtuple('Setting', ['activations', 'n_chip', 'mem_size'])
Result = namedtuple('Result', ['msg_cnt', 'tot_mem', 'avg_act_size'])

random.seed(4)

n_chips = [i for i in range(8, MAX_N_CHIP, 16)]
activations_list = [gen_model() for _ in range(N_MODEL)]
activations_list[0] = [26, 19, 4, 15]

for acts in activations_list:
    print("activations: ", acts, flush=True)
    for n_chip in n_chips:
        print("#chip: ", n_chip, flush=True)
        kahypar_partition_msg(acts, n_chip)
exit()

Settings = [Setting(act, n_chip, get_mem_size(act, n_chip, RELAXED_RATIO)) for act in activations_list for n_chip in n_chips]
Results = [Result(cnt_partition_msg(setting.activations, setting.n_chip, setting.mem_size), get_tot_mem(setting.activations), get_avg_act_size(setting.activations)) for setting in Settings]


for i in range(N_MODEL):
    avg_act_size = [res.avg_act_size for res in Results[i*MAX_N_CHIP:(i+1)*MAX_N_CHIP]]
    msg_cnt = [res.msg_cnt for res in Results[i*MAX_N_CHIP:(i+1)*MAX_N_CHIP]]
    plt.ylabel('msg_cnt')
    plt.xlabel('n_chip')
    plt.plot(n_chips, avg_act_size)
    plt.plot(n_chips, msg_cnt, 'ro')
    plt.show()