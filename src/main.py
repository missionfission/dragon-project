import glob
import os
import sys
from collections import deque
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import yaml
import yamlordereddictloader
from torchvision import models
from yaml import dump

from dlrm.dlrm_s_pytorch import DLRM_Net, dash_separated_floats, dash_separated_ints
from generator import *
from generator import Generator, get_compute_props, get_mem_props
from ir.handlers import handlers
from ir.trace import get_backprop_memory, trace
from scheduling import Scheduling
from utils.logger import create_logger
from utils.visualizer import *
from utils.visualizer import plot_descent


####################################
def design_tech_runner(graph_set, backprop=False, print_stats=False):
    """
    Runs the Input Graph
    """
    generator = Generator()
    bandwidth = [2, 10, 50, 75, 100]
    num_iterations = 50
    for graph in graph_set:
        scheduler = Scheduling()
        (
            read_bw_req,
            write_bw_req,
            read_bw_actual,
            write_bw_actual,
            cycles,
            free_cycles,
        ) = scheduler.run(graph)
        read_bw_limit, write_bw_limit = (
            scheduler.mem_read_bw[scheduler.mle - 1],
            scheduler.mem_write_bw[scheduler.mle - 1],
        )
        #         bandwidth_bar_graph("read_full.png", read_bw_req, read_bw_actual, read_bw_limit, graph.nodes)
        #         cycles_bar_graph("cycles.png", cycles, free_cycles, graph.nodes)
        #         mem_util_bar_graph("mem_util.png",scheduler.mem_util_full/scheduler.mem_size[0],scheduler.mem_util_log/scheduler.mem_size[0], graph.nodes)
        #         in_time, in_energy, design, tech = generator.save_stats(scheduler)
        in_time, in_energy, design, tech = generator.save_stats(
            scheduler, backprop, get_backprop_memory(graph.nodes), print_stats
        )
        print("======Optimizing Design and Connectivity=========")
        for i in range(num_iterations):
            config = generator.backward_pass(scheduler)
            generator.writeconfig(config, str(i) + "hw.yaml")
            scheduler.create_config(config)
            _, _, _, _, cycles, free_cycles = scheduler.run(graph)
            time, energy, design, tech = generator.save_stats(
                scheduler, backprop, get_backprop_memory(graph.nodes), print_stats
            )
        print(in_time[0] // time[0], in_energy[0] // energy[0])
        print("===============Optimizing Technology=============")
        for i in range(10):
            config = generator.backward_pass_tech(scheduler, "time")
            generator.writeconfig(config, str(i) + "hw.yaml")
            scheduler.create_config(config)
            _, _, _, _, cycles, free_cycles = scheduler.run(graph)
            time, energy, design, tech = generator.save_stats(
                scheduler, backprop, get_backprop_memory(graph.nodes), print_stats
            )
        print(in_time[0] // time[0], in_energy[0] // energy[0])


def runner_save_all(graph_set, scheduler, backprop=False, print_stats=False):
    """
    Runs the Input Graph
    """
    time_list = []
    energy_list = []
    bandwidth_time_list = []
    mem_size_idle_time_list = []
    bank_list = []
    mem_size_list = []
    compute_list = []
    tech_params_list = []
    num_iterations = 6
    generator = Generator()
    bandwidth = [2, 10, 50, 75, 100]
    for graph in graph_set:
        (
            read_bw_req,
            write_bw_req,
            read_bw_actual,
            write_bw_actual,
            cycles,
            free_cycles,
        ) = scheduler.run(graph)
        read_bw_limit, write_bw_limit = (
            scheduler.mem_read_bw[scheduler.mle - 1],
            scheduler.mem_write_bw[scheduler.mle - 1],
        )
        #         bandwidth_bar_graph("read_full.png", read_bw_req, read_bw_actual, read_bw_limit, graph.nodes)
        #         cycles_bar_graph("cycles.png", cycles, free_cycles, graph.nodes)
        #         mem_util_bar_graph("mem_util.png",scheduler.mem_util_full/scheduler.mem_size[0],scheduler.mem_util_log/scheduler.mem_size[0], graph.nodes)
        time, energy, design, tech = generator.save_stats(
            scheduler, backprop, get_backprop_memory(graph.nodes), print_stats
        )
        time_list.append(time[0])
        energy_list.append(energy[0])
        bandwidth_time_list.append(time[1])
        mem_size_idle_time_list.append(time[2])
        bank_list.append(design[0])
        mem_size_list.append(design[1])
        tech_params_list.append(tech)
        #         print(scheduler.config)
        for i in range(num_iterations):
            config = generator.backward_pass(scheduler, "time")
            generator.writeconfig(config, str(i) + "hw.yaml")
            scheduler.create_config(config)
            (
                read_bw_req,
                write_bw_req,
                read_bw_actual,
                write_bw_actual,
                cycles,
                free_cycles,
            ) = scheduler.run(graph)
            #             read_bw_limit, write_bw_limit = scheduler.mem_read_bw[scheduler.mle - 1], scheduler.mem_write_bw[scheduler.mle - 1]
            # #             bandwidth_bar_graph("read_full.png", read_bw_req, read_bw_actual, read_bw_limit, graph.nodes, cycles)
            time, energy, design, tech = generator.save_stats(
                scheduler, backprop, get_backprop_memory(graph.nodes), print_stats
            )
            time_list.append(time[0])
            energy_list.append(energy[0])
            bandwidth_time_list.append(time[1])
            mem_size_idle_time_list.append(time[2])
            bank_list.append(design[0])
            mem_size_list.append(design[1])
            tech_params_list.append(tech)
        return [
            time_list,
            bandwidth_time_list,
            mem_size_idle_time_list,
            bank_list,
            mem_size_list,
            compute_list,
            tech_params_list,
        ]


def design_runner(graph_set, scheduler, backprop=False, print_stats=False):
    """
    Runs the Input Graph
    """
    generator = Generator()
    bandwidth = [2, 10, 50, 75, 100]
    num_iterations = 50
    for graph in graph_set:
        (
            read_bw_req,
            write_bw_req,
            read_bw_actual,
            write_bw_actual,
            cycles,
            free_cycles,
        ) = scheduler.run(graph)
        read_bw_limit, write_bw_limit = (
            scheduler.mem_read_bw[scheduler.mle - 1],
            scheduler.mem_write_bw[scheduler.mle - 1],
        )
        #         bandwidth_bar_graph("read_full.png", read_bw_req, read_bw_actual, read_bw_limit, graph.nodes)
        #         cycles_bar_graph("cycles.png", cycles, free_cycles, graph.nodes)
        #         mem_util_bar_graph("mem_util.png",scheduler.mem_util_full/scheduler.mem_size[0],scheduler.mem_util_log/scheduler.mem_size[0], graph.nodes)
        in_time, in_energy, design, tech = generator.save_stats(
            scheduler, backprop, get_backprop_memory(graph.nodes)
        )
        print("======Optimizing Design and Connectivity=========")
        for i in range(num_iterations):
            config = generator.backward_pass(scheduler)
            generator.writeconfig(config, str(i) + "hw.yaml")
            scheduler.create_config(config)
            _, _, _, _, cycles, free_cycles = scheduler.run(graph)
            time, energy, design, tech = generator.save_stats(
                scheduler, backprop, get_backprop_memory(graph.nodes), print_stats
            )
        print(in_time[0] // time[0], in_energy[0] // energy[0])
