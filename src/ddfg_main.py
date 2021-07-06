import glob
import os

####################################
import subprocess
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
from synthesis import hls
from ddfg_scheduling import DDFG_Scheduling
from generator import *
from generator import Generator, get_mem_props
from ir.handlers import handlers
from ir.trace import trace
from ir.staticfg.staticfg import CFGBuilder

from utils.visualizer import *
from utils.visualizer import (
    bandwidth_bar_graph,
    cycles_bar_graph,
    mem_util_bar_graph,
    plot_gradients,
)


def run_mapping(scheduler, mapping, graph):
    if mapping == "asap":
        scheduler.run_asap(graph)
    elif mapping == "reuse_full":
        scheduler.run_reuse_full(graph)
    elif mapping == "reuse_leakage":
        scheduler.run_reuse_leakage(graph)


def synthesis_hardware(benchmark):
    if benchmark == "aes":
        bashCommand = "common/aladdin aes  aes_aes/inputs/dynamic_trace.gz aes_aes/test_aes.cfg"
        process = subprocess.Popen(
            bashCommand.split(), stdout=subprocess.PIPE, cwd="./req/"
        )
        output, error = process.communicate()
        for i in output.decode("utf-8").split("\n"):
            print(i)
    if benchmark == "bfs":
        bashCommand = "common/aladdin bfs_bulk  bfs_bulk/inputs/dynamic_trace1.gz bfs_bulk/test_bfs.cfg"
        process = subprocess.Popen(
            bashCommand.split(), stdout=subprocess.PIPE, cwd="./req/"
        )
        output, error = process.communicate()
        for i in output.decode("utf-8").split("\n"):
            print(i)
    if benchmark == "hpcg":
        cfg = CFGBuilder().build_from_file(
            "hpcg.py",
            "nonai_models/hpcg.py",
        )
        hls.parse_graph(cfg)
        hls.get_stats(cfg)
        

####################################
def design_tech_runner(
    graph_set,
    backprop=False,
    print_stats=False,
    file="default.yaml",
    stats_file="logs/stats.txt",
):
    """
    Runs the Input Graph
    """

    num_iterations = 50
    for graph in graph_set:
        generator = Generator()
        scheduler = DDFG_Scheduling(stats_file=stats_file)
        scheduler.run_asap(graph)
        in_time, in_energy, design, tech, in_area = generator.save_stats(
            scheduler, backprop, get_backprop_memory(graph.nodes), print_stats
        )
        scheduler = DDFG_Scheduling(stats_file=stats_file)
        print("======Optimizing Design and Connectivity=========")
        i = 0
        while True:
            _, _, _, _, cycles, free_cycles = scheduler.run_asap(graph)
            time, energy, design, tech, area = generator.save_stats(
                scheduler, backprop, get_backprop_memory(graph.nodes), print_stats
            )
            if (
                scheduler.bandwidth_idle_time < 0.1 * scheduler.total_cycles
                or scheduler.force_connectivity
            ) and scheduler.mem_size_idle_time < 0.1 * scheduler.total_cycles:
                break
            # print(area / in_area)
            config = generator.backward_pass_design(scheduler)
            generator.writeconfig(config, str(i) + "hw.yaml")
            scheduler.complete_config(config)
            i += 1

        print(in_time[0] / time[0], in_energy[0] / energy[0], in_area[0] / area[0])
        print("===============Optimizing Technology=============")
        for j in range(10):
            _, _, _, _, cycles, free_cycles = scheduler.run_asap(graph)
            time, energy, design, tech, area = generator.save_stats(
                scheduler, backprop, get_backprop_memory(graph.nodes), print_stats
            )
            config = generator.backward_pass_tech(scheduler, "time")
            generator.writeconfig(config, str(j + i) + "hw.yaml")
            scheduler.complete_config(config)
        print(in_time[0] / time[0], in_energy[0] / energy[0])
    return time, energy, area


def design_runner(
    graph_set,
    backprop=False,
    print_stats=False,
    file="default.yaml",
    stats_file="logs/stats.txt",
):
    """
    Runs the Input Graph
    """

    time_list = []
    energy_list = []
    design_list = []
    tech_list = []
    num_iterations = 50
    for graph in graph_set:

        # (
        #     read_bw_req,
        #     write_bw_req,
        #     read_bw_actual,
        #     write_bw_actual,
        #     cycles,
        #     free_cycles,
        # ) = scheduler.run_asap(graph)
        # read_bw_limit, write_bw_limit = (
        #     scheduler.mem_read_bw[scheduler.mle - 1],
        #     scheduler.mem_write_bw[scheduler.mle - 1],
        # )
        #         bandwidth_bar_graph("read_full.png", read_bw_req, read_bw_actual, read_bw_limit, graph.nodes)
        #         cycles_bar_graph("cycles.png", cycles, free_cycles, graph.nodes)
        #         mem_util_bar_graph("mem_util.png",scheduler.mem_util_full/scheduler.mem_size[0],scheduler.mem_util_log/scheduler.mem_size[0], graph.nodes)
        generator = Generator()
        scheduler = DDFG_Scheduling(hwfile=file, stats_file=stats_file)
        scheduler.run_asap(graph)
        in_time, in_energy, in_design, in_tech, in_area = generator.save_stats(
            scheduler, backprop, get_backprop_memory(graph.nodes), print_stats
        )
        scheduler = DDFG_Scheduling(hwfile=file, stats_file=stats_file)
        i = 0
        print("======Optimizing Design=========")
        while True:
            _, _, _, _, cycles, free_cycles = scheduler.run_asap(graph)
            time, energy, design, tech, area = generator.save_stats(
                scheduler, backprop, get_backprop_memory(graph.nodes), print_stats
            )
            if (
                scheduler.bandwidth_idle_time < 0.1 * scheduler.total_cycles
                or scheduler.force_connectivity
            ) and scheduler.mem_size_idle_time < 0.1 * scheduler.total_cycles:
                break
            # print(area / in_area)
            config = generator.backward_pass_design(scheduler)
            generator.writeconfig(config, str(i) + "hw.yaml")
            scheduler.complete_config(config)
            time_list.append(time)
            energy_list.append(energy)
            design_list.append(design)
            i += 1
        print(in_time[0] / time[0], in_energy[0] / energy[0], in_area / area)

    return time, energy, area
    # return time_list, energy_list, design_list


