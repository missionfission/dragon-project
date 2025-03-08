"""DragonX: Hardware-Software Co-Design Framework for AI Accelerators

This module provides the main interfaces for:
- Workload analysis and profiling
- Neural network optimization 
- Auto-tuning and parameter optimization
- Performance prediction and modeling
- Compiler optimizations

Example usage:
    import src_main as dx
    
    # Initialize optimizer with architecture config
    optimizer = dx.initialize(arch_config="custom_accelerator.yaml") 

    # Define and analyze workload
    graph = dx.analyze_workload(model)
    
    # Optimize design for target metrics
    optimized_config = dx.optimize_design(
        graph,
        target_metrics={
            "latency": "minimal",
            "power": "<5W"
        }
    )

    # Get performance estimates
    perf_stats = dx.estimate_performance(graph, optimized_config)
"""

import glob
import os
import sys
from collections import deque
from copy import deepcopy
import io
import base64
import numpy as np
from yaml import dump
from generator import *
from generator import Generator, get_mem_props
from ir.handlers import handlers  
from ir.trace import get_backprop_memory, trace
from mapper.mapper import Mapper
from utils.visualizer import (
    bandwidth_bar_graph,
    cycles_bar_graph, 
    mem_util_bar_graph,
    plot_gradients
)
import matplotlib.pyplot as plt
import math

def initialize(arch_config="default.yaml", **kwargs):
    """Initialize the DragonX framework.
    
    Args:
        arch_config (str): Path to architecture configuration YAML
        **kwargs: Additional configuration parameters
        
    Returns:
        Generator: Configured optimization framework instance
    """
    return Generator(hwfile=arch_config, **kwargs)

def analyze_workload(model, backprop=False):
    """Analyze a workload/model to generate computation graph.
    
    Args:
        model: PyTorch model or other workload to analyze
        backprop (bool): Whether to include backpropagation
        
    Returns:
        Graph: Computation graph representation
    """
    return trace(model, backprop)

def optimize_design(graph, target_metrics, backprop=False, **kwargs):
    """Optimize hardware design for target metrics.
    
    Args:
        graph: Computation graph to optimize for
        target_metrics (dict): Target performance metrics
        **kwargs: Additional optimization parameters
        
    Returns:
        dict: Optimized hardware configuration
    """
    generator = Generator()
    mapper = Mapper()
    
    # Run optimization
    time, energy, design, tech, area = generator.save_stats(
        mapper, backprop, get_backprop_memory(graph.nodes)
    )
    
    # Optimize based on target metrics
    config = generator.backward_pass_design(mapper)
    
    return config

def estimate_performance(graph, config, backprop=False, print_stats=True):
    """Estimate performance of graph on given hardware config.
    
    Args:
        graph: Computation graph to evaluate
        config (dict): Hardware configuration
        backprop (bool): Whether to include backpropagation
        print_stats (bool): Whether to print detailed stats
        
    Returns:
        tuple: (execution_time, energy, area) performance estimates
    """
    mapper = Mapper(hwfile=config)
    mapper.run_asap(graph)
    
    generator = Generator()
    time, energy, design, tech, area = generator.save_stats(
        mapper, backprop, get_backprop_memory(graph.nodes), print_stats
    )
    
    return time, energy, area

# Other existing functions...

def get_design_points_area_scaled(area_budget, connectivity, node, pitch):
    
    connectivity_area = connectivity/pitch^2
    accel_area = area_budget - connectivity_area
    
    # memory configuration, area scaling, buffer sizing happening in separate functions
    # converge them here
        
    # connectivity scale memories
    
    print("Buffer size", pe_width*pe_count*2)
    print("Memory Configuration", sram_banks)
    print("Memory Configuration", sram_size)
    print("No of PEs", pe_count)
    pass
          
def design_runner(
    graph_set,
    backprop=False,
    print_stats=False,
    file="default.yaml",
    stats_file="logs/stats.txt",
):
    """Runs the Input Graph and Optimizes Design 

    Args:
        graph_set (): 
        backprop (bool, optional): . Defaults to False.
        print_stats (bool, optional): . Defaults to False.
        file (str, optional): . Defaults to "default.yaml".
        stats_file (str, optional): . Defaults to "logs/stats.txt".

    Returns:
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
        # ) = mapper.run_asap(graph)
        # read_bw_limit, write_bw_limit = (
        #     mapper.mem_read_bw[mapper.mle - 1],
        #     mapper.mem_write_bw[mapper.mle - 1],
        # )
        #         bandwidth_bar_graph("read_full.png", read_bw_req, read_bw_actual, read_bw_limit, graph.nodes)
        #         cycles_bar_graph("cycles.png", cycles, free_cycles, graph.nodes)
        #         mem_util_bar_graph("mem_util.png",mapper.mem_util_full/mapper.mem_size[0],mapper.mem_util_log/mapper.mem_size[0], graph.nodes)
        generator = Generator()
        mapper = Mapper(hwfile=file, stats_file=stats_file)
        mapper.run_asap(graph)
        in_time, in_energy, in_design, in_tech, in_area = generator.save_stats(
            mapper, backprop, get_backprop_memory(graph.nodes), print_stats
        )
        mapper = Mapper(hwfile=file, stats_file=stats_file)
        i = 0
        print("======Optimizing Design=========")
        while True:
            _, _, _, _, cycles, free_cycles = mapper.run_asap(graph)
            time, energy, design, tech, area = generator.save_stats(
                mapper, backprop, get_backprop_memory(graph.nodes), print_stats
            )
            if (
                mapper.bandwidth_idle_time < 0.1 * mapper.total_cycles
                or mapper.force_connectivity
            ) and mapper.mem_size_idle_time < 0.1 * mapper.total_cycles:
                break
            # print(area / in_area)
            config = generator.backward_pass_design(mapper)
            generator.writeconfig(config, str(i) + "hw.yaml")
            mapper.complete_config(config)
            time_list.append(time)
            energy_list.append(energy)
            design_list.append(design)
            i += 1
        print(
            "Faster : ",
            in_time[0] / time[0],
            "Energy Improvement : ",
            in_energy[0] / energy[0],
            "Area Budget : ",
            in_area / area,
        )

    return time, energy, area
#     return time_list, energy_list, design_list, area


def design_tech_runner(
    graph_set,
    backprop=False,
    print_stats=False,
    file="default.yaml",
    stats_file="logs/stats.txt",
):
    """[Runs the Input Graph : Optimizes Design and Technology]

    Args:
        graph_set (): 
        backprop (bool, optional): . Defaults to False.
        print_stats (bool, optional): . Defaults to False.
        file (str, optional): . Defaults to "default.yaml".
        stats_file (str, optional): . Defaults to "logs/stats.txt".

    Returns:
        : 
    """
    num_iterations = 50
    for graph in graph_set:
        generator = Generator()
        mapper = Mapper(stats_file=stats_file)
        mapper.run_asap(graph)
        in_time, in_energy, design, tech, in_area = generator.save_stats(
            mapper, backprop, get_backprop_memory(graph.nodes), print_stats
        )
        mapper = Mapper(stats_file=stats_file)
        ("======Optimizing Design and Connectivity=========")
        i = 0
        while True:
            _, _, _, _, cycles, free_cycles = mapper.run_asap(graph)
            time, energy, design, tech, area = generator.save_stats(
                mapper, backprop, get_backprop_memory(graph.nodes), _stats
            )
            if (
                mapper.bandwidth_idle_time < 0.1 * mapper.total_cycles
                or mapper.force_connectivity
            ) and mapper.mem_size_idle_time < 0.1 * mapper.total_cycles:
                break
            # (area / in_area)
            config = generator.backward_pass_design(mapper)
            generator.writeconfig(config, str(i) + "hw.yaml")
            mapper.complete_config(config)
            i += 1

        (in_time[0] / time[0], in_energy[0] / energy[0], in_area[0] / area[0])
        print("===============Optimizing Technology=============")
        for j in range(10):
            _, _, _, _, cycles, free_cycles = mapper.run_asap(graph)
            time, energy, design, tech, area = generator.save_stats(
                mapper, backprop, get_backprop_memory(graph.nodes), print_stats
            )
            config = generator.backward_pass_tech(mapper, "time")
            generator.writeconfig(config, str(j + i) + "hw.yaml")
            mapper.complete_config(config)
        print(in_time[0] / time[0], in_energy[0] / energy[0])
    return time, energy, area


def perf(
    graph, backprop, print_stats, filename, mapping="nn_dataflow", *args, **kwargs
):
    """

    Args:
        graph (): 
        backprop (): 
        print_stats (): 
        filename (): 
        mapping (str, optional): . Defaults to "nn_dataflow".

    Returns:
        : 
    """
    mapper = Mapper(hwfile=filename)
    if mapping == "asap":
        mapper.run_asap(graph)
    elif mapping == "nn_dataflow":
        mapper.run_nn_dataflow(graph)
    elif mapping == "reuse_full":
        mapper.run_reuse_full(graph)
    elif mapping == "reuse_leakage":
        mapper.run_reuse_leakage(graph)
    generator = Generator()
    in_time, in_energy, design, tech, area = generator.save_stats(
        mapper, backprop, get_backprop_memory(graph.nodes), print_stats
    )
    return in_time, in_energy, area


def all_design_updates(graph, backprop):
    """Plots the Design Parameters Updates in Backward Pass on Running a Given DFG
    Args:
        graph (): 
        backprop (): True to Run the Workload in Training
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    # ax2 = ax.twinx()
    base_dir = "figures/"
    design_list = []
    design_names = []
    time_list = []
    energy_list = []
    mapper = Mapper()
    mapper.run_asap(graph)
    generator = Generator()
    print_stats = True
    in_time, in_energy, in_design, in_tech, in_area = generator.save_stats(
        mapper, backprop, get_backprop_memory(graph.nodes), print_stats
    )
    for i in range(num_iterations):
        config = generator.backward_pass_design(mapper)
        generator.writeconfig(config, str(i) + "hw.yaml")
        mapper.complete_config(config)
        _, _, _, _, cycles, free_cycles = mapper.run_asap(graph)
        time, energy, design, tech, area = generator.save_stats(
            mapper, backprop, get_backprop_memory(graph.nodes), print_stats
        )
        design_list.append(design)
        time_list.append(time)
        energy_list.append(energy)
    design_list = np.array(design_list)
    time_list = np.array(time_list)
    energy_list = np.array(energy_list)
    for i in range(len(in_design)):
        ax.plot(design_list[:, i] / in_design[i], label=design_names[i])
    ax.plot(time_list[:, 0] / in_time[0], label="Execution Time")
    ax.plot(energy_list[:, 0] / in_energy[0], label="Energy Consumption")


def all_tech_updates(graph, backprop):
    """Plots the Technology Parameters Updates in Backward Pass on Running a Given DFG
    Args:
        graph (): 
        backprop (): 
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    # ax2 = ax.twinx()
    tech_names = []
    tech_list = []
    time_list = []
    energy_list = []
    base_dir = "figures/"
    mapper = Mapper()
    mapper.run_asap(graph)
    generator = Generator()
    print_stats = True
    in_time, in_energy, in_design, in_tech, in_area = generator.save_stats(
        mapper, backprop, get_backprop_memory(graph.nodes), print_stats
    )
    for i in range(num_iterations):
        config = generator.backward_pass_design(mapper)
        generator.writeconfig(config, str(i) + "hw.yaml")
        mapper.complete_config(config)
        _, _, _, _, cycles, free_cycles = mapper.run_asap(graph)
        time, energy, design, tech, area = generator.save_stats(
            mapper, backprop, get_backprop_memory(graph.nodes), print_stats
        )
    for i in range(10):
        config = generator.backward_pass_tech(mapper, "time")
        generator.writeconfig(config, str(i) + "hw.yaml")
        mapper.complete_config(config)
        _, _, _, _, cycles, free_cycles = mapper.run_asap(graph)
        time, energy, design, tech, area = generator.save_stats(
            mapper, backprop, get_backprop_memory(graph.nodes), print_stats
        )
        time_list.append(time)
        energy_list.append(energy)
        tech_list.append(tech)
    time_list = np.array(time_list)
    energy_list = np.array(energy_list)
    tech_list = np.array(tech_list)

    for i in len(in_tech):
        ax.plot(tech_list[:, i] / in_tech[i], label=tech_names[i])
    ax.plot(time_list[:, 0] / in_time[0], label="Execution Time")
    ax.plot(energy_list[:, 0] / in_energy[0], label="Energy Consumption")


def s_mem_c_same_arch(
    graph_list, backprop, names=None, plot="time", area_budget=2.5, *args, **kwargs
):
    """
    Fix Everything in Architecture and Just Sweep Memory Connectivity
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    for en, graph in enumerate(graph_list):
        time_list = []
        energy_list = []
        for j in range(1, 4000, 4):
            mapper = Mapper(hwfile="illusion.yaml")
            generator = Generator()
            mapper.config["memory"]["level1"]["banks"] = 1
            mapper.config["memory"]["level1"]["banks"] *= j
            if names[en] == "SSD":
                mapper.config["memory"]["level0"]["size"] *= 6
            mapper.complete_config(mapper.config)
            mapper.run_asap(graph)
            in_time, in_energy, design, tech, area = generator.save_stats(
                mapper, backprop, get_backprop_memory(graph.nodes)
            )
            time_list.append(in_time[0])
            energy_list.append(in_energy[0])
        if plot == "time":
            if names[en] == "DLRM":
                time_list = np.array(time_list) * 50
            ax.plot(np.arange(2, 8000, 8), time_list, "o-", label=names[en])
        elif plot == "energy":
            ax.plot(energy_list, "o-", label=names[en])
        else:
            # plot edp
            ax.plot(
                [x * energy_list[enum] for enum, x in enumerate(time_list)],
                "o-",
                label=names[en],
            )
    # ax.plot()
    ax.set_xlabel("Memory Connectivity", fontsize=20, fontweight="bold")
    ax.set_ylabel("EDP", fontsize=20, fontweight="bold")
    plt.rc("xtick", labelsize=20)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=20)
    xposition = [1, 8, 256, 25600]
    names = ["DRAM", "5um Pitch", "1um Pitch", "100nm Pitch"]
    colors = ["r", "b", "g", "k"]
    for i, xc in enumerate(xposition):
        plt.axvline(x=xc, label=names[i], color=colors[i], linestyle="--")
    ax.legend(fontsize=20)
    plt.yscale("log")
    plt.xscale("log")
    fig.tight_layout()
    plt.savefig(
        "figures/connectivity_sweep_area" + str(plot) + str(area_budget) + ".png",
        bbox_inches="tight",
    )
    plt.show()


def s_size_c_joint(graph, backprop):
    """
    Change Memory Connectivity and Memory Size in Conjuction see how those two are correlated
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    for en, graph in enumerate(graph):
        time_list = []
        energy_list = []
        for j in range(1, 100):
            mapper = Mapper()
            generator = Generator()
            mapper.config["memory"]["level1"]["banks"] = 2
            mapper.config["memory"]["level1"]["banks"] *= j
            mapper.complete_config(mapper.config)
            mapper.run_asap(graph)
            # mapper.config["memory"]["level0"]["size"] *= 2
            in_time, in_energy, design, tech = generator.save_stats(
                mapper, backprop, get_backprop_memory(graph.nodes)
            )
            time_list.append(in_time[0])
            energy_list.append(in_energy[0])
        ax.plot(energy_list, "o-")
        ax.plot(time_list, "o-")
    ax.set_xlabel("Memory Connectivity", fontsize=20, fontweight="bold")
    ax.set_ylabel("Energy Consumption", fontsize=20, fontweight="bold")
    plt.rc("xtick", labelsize=20)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=20)
    ax.legend(fontsize=20)
    plt.yscale("log")
    fig.tight_layout()
    plt.savefig("figures/connectivity_sweep_energy.png", bbox_inches="tight")
    plt.show()


def sweep(graph_list, backprop, names=None, plot="time", area_range=1, *args, **kwargs):
    """
    Sweeps over Connectivity, Precision, Nodes (nm), Area Budget and Uniform/Non Uniform Memory Configurations.
    
    Args:
        graph_list (): List of AI/Non AI Workloads to Run
        backprop (): 
        names (, optional): . Defaults to None.
        plot (str, optional): . Defaults to "time".
        area_range (int, optional): . Defaults to 1.
    """
    total_mem = show_memory_capacity_req(graph_list, backprop, names=names, plot="time")
    fig, ax = plt.subplots(figsize=(6, 6))

    area_names = ["Edge", "Mid", "Cloud", "Giant", "Wafer-Scale"]
    markers_plot = ["o", "+", "x", "o"]
    colors = {
        "Resnet18": "r",
        "SLAM": "mediumseagreen",
        "SSD": "g",
        "BERT": "k",
        "Genomics": "c",
        "RNN": "aquamarine",
        "PageRank": "y",
        "DLRM": "m",
        "Resnet50": "b",
    }
    precision = [32, 8, 4]
    precision_density_factor = [1, 16, 64]
    precision_speed_factor = [1, 4, 8]
    precision_power_factor = [1, 1 / (0.7), 1 / 0.7]
    precision_size_factor = [1, 4, 8]
    node_name = [28, 7, 3]
    node_speed = [1, 2, 3]
    node_density = [1, 6, 14]
    node_energy = [1, 0.42, 0.24]
    m = 2
    node = 3

    for en, graph in enumerate(graph_list):
        p = 0
        if names[en] in ["Resnet18", "Resnet50", "SSD"]:
            p = 2
        temp = 1

        # for p in range(3):
        print(names[en], p)
        for area_budget in range(area_range):
            if names[en] in ["DLRM", "PageRank", "Genomics"]:
                area_budget = 2
            energy_list = []
            energy_list2 = []
            # connectivity = 2*j*32
            # for m, node in enumerate(node_name):
            pitch_list = []
            pitch_list2 = []
            # if names[en] == "PageRank":
            #     pitch_list = [
            #         10000000,
            #         1300000,
            #         400000,
            #         140000,
            #         70000,
            #         20000,
            #         8000,
            #     ]
            # elif names[en] == "Genomics":
            #     pitch_list = [
            #         10000000,
            #         900000,
            #         400000,
            #         200000,
            #         100000,
            #         70000,
            #         50000,
            #     ]
            # elif names[en] == "SLAM":
            #     pitch_list = [
            #         10000000,
            #         1600000,
            #         600000,
            #         180000,
            #         90000,
            #         40000,
            #         20000,
            #     ]
            # else:
            for i, pitch in enumerate([10, 5, 3, 2, 1, 0.5, 0.1]):
                percent_time_list = []
                percent_time_list2 = []
                for percent in range(5, 95, 10):

                    connectivity_area = (
                        percent * 1000 * 10 ** area_budget / node_density[m]
                    )
                    j = connectivity_area / (2 * 32 * pitch ** 2) / total_mem[en]
                    mapper = Mapper(hwfile="illusion.yaml")
                    generator = Generator()
                    mapper.config["mm_compute"]["N_PE"] *= (
                        10 ** area_budget * precision_density_factor[p]
                    )
                    mapper.config["mm_compute"]["frequency"] *= (
                        node_speed[m] * precision_speed_factor[p]
                    )
                    mapper.config["mm_compute"]["per_op_energy"] *= node_energy[m]

                    mapper.config["memory"]["level0"]["size"] *= (
                        10 ** area_budget * precision_size_factor[p]
                    )
                    mapper.config["memory"]["level1"]["banks"] = 2
                    mapper.config["memory"]["level1"]["banks"] *= j
                    mapper.complete_config(mapper.config)
                    mapper.run_asap(graph)
                    print_stats = False
                    if percent == 95:
                        print_stats = True
                    (
                        in_time,
                        in_energy,
                        design,
                        tech,
                        area,
                    ) = generator.save_stats(
                        mapper, backprop, get_backprop_memory(graph.nodes), print_stats
                    )
                    
                    new_area = area + connectivity_area
                    percent_time_list2.append(in_time[0])
                    energy_list2.append(in_energy[0])

                    j2 = connectivity_area / (2 * 32 * pitch ** 2)
                    mapper = Mapper(hwfile="illusion.yaml")
                    generator = Generator()
                    mapper.config["mm_compute"]["N_PE"] *= (
                        10 ** area_budget * precision_density_factor[p]
                    )
                    mapper.config["mm_compute"]["frequency"] *= (
                        node_speed[m] * precision_speed_factor[p]
                    )
                    mapper.config["mm_compute"]["per_op_energy"] *= node_energy[m]

                    mapper.config["memory"]["level0"]["size"] *= (
                        10 ** area_budget * precision_size_factor[p]
                    )
                    mapper.config["memory"]["level1"]["banks"] = 2
                    mapper.config["memory"]["level1"]["banks"] *= j2
                    mapper.complete_config(mapper.config)
                    mapper.run_asap(graph)
                    (
                        in_time,
                        in_energy,
                        design,
                        tech,
                        area,
                    ) = generator.save_stats(
                        mapper, backprop, get_backprop_memory(graph.nodes)
                    )
                    new_area = area / 17 + connectivity_area
                    percent_time_list.append(in_time[0])
                    energy_list.append(in_energy[0])

                pitch_list.append(
                    min(percent_time_list)
                    * node_energy[m]
                    * 1
                    / (precision_power_factor[p] * 0.3)
                )
                pitch_list2.append(
                    min(percent_time_list2)
                    * node_energy[m]
                    * 1
                    / (precision_power_factor[p] * 0.3)
                )
                # np.arange(2,8000,8)
                # temp = pitch_list2[0]
                # for i in range(len(pitch_list)):
                #     pitch_list2[i] = temp / pitch_list2[i]
                # print(pitch_list)

            if m == 0:
                # if area_budget == 0:
                temp = pitch_list[0]
            for i in range(len(pitch_list)):
                pitch_list[i] = temp / pitch_list[i]
            if plot == "time":
                # ax.plot(['10','5','3','2','1','0.5','0.1'],pitch_list, color = colors[en],marker=markers_plot[area_budget], label=area_names[area_budget]+names[en] + "precision : "+ str(precision))
                if precision[p] == 32:
                    # labels = names[en] + ":" + str(node) + "nm"
                    # + ":fp32"
                    labels = names[en]
                    # labels = names[en] + ":" + area_names[area_budget]
                    # labels = names[en] + ":" ":Uniform"
                    # # # :fp32
                    # labels2 = names[en] + ":Non-Uniform"
                    # :fp32
                else:
                    # labels = names[en] + ":" + str(node) + "nm"
                    labels = names[en]
                    # labels = names[en] + ":" + area_names[area_budget]
                    # # ":int" + str(precision[p])
                    # labels = names[en] + ":Uniform"

                    # labels2 = names[en] + ":Non-Uniform"
                    # ":int" + str(precision[p])
                    # labels = (
                    #     names[en]
                    #     + ":"
                    #     + str(node)
                    #     + "nm"
                    #     + ":"
                    #     + "int"
                    #     + str(precision[p])
                    # )
                    # labels = (
                    #     names[en]
                    #     + ":"
                    #     + area_names[area_budget]
                    #     + ":"
                    #     + "int"
                    #     + str(precision[p])
                    # )
                    # labels2 = (
                    #     area_names[area_budget]
                    #     + names[en]
                    #     + "int"
                    #     + str(precision[p])
                    #     + " "
                    #     + str(node)
                    #     + "nm"
                    #     + "NU"
                    # )
                # ax.bar(
                #     3.8 * np.arange(7) + en * 0.4 + 1.2 * area_budget,
                #     pitch_list,
                #     width=0.4,
                #     color=colors[en],
                #     hatch=markers_plot[area_budget],
                #     label=labels,
                # )

                # ax.bar(
                #     3.8 * np.arange(7) + en * 0.4 + 1.2 * area_budget + 0.5,
                #     pitch_list2,
                #     width=0.4,
                #     color=colors[en],
                #     hatch="+",
                #     label=labels2,
                # )
                ax.plot(
                    [10, 5, 3, 2, 1, 0.5, 0.1],
                    pitch_list,
                    color=colors[names[en]],
                    marker=markers_plot[m],
                    label=labels,
                    linewidth=2,
                )
                # ax.plot(
                #     [10, 5, 3, 2, 1, 0.5, 0.1],
                #     pitch_list,
                #     color=colors[names[en]],
                #     marker=markers_plot[area_budget],
                #     label=labels,
                #     linewidth=2,
                # )
                # ax.plot(
                #     [10, 5, 3, 2, 1, 0.5, 0.1],
                #     pitch_list2,
                #     color=colors[names[en]],
                #     marker=markers_plot[m + 1],
                #     label=labels2,
                #     linewidh=2,
                # )
                print(pitch_list)

                # ax.plot(
                #     pitch_list,
                #     "o-",
                #     color=colors[en],
                #     marker=markers_plot[m],
                #     label=labels,
                # )
                # ['10','5','3','2','1','0.5','0.1']
                # ax.plot(['10','5','3','2','1','0.5','0.1'],pitch_list, color = colors[en],, label=area_names[area_budget]+names[en] + " fp32")
    ax.set_xlabel("Vertical Interconnect Pitch (in um)", fontsize=20, fontweight="bold")
    ax.set_ylabel(
        "EDP Benefit \n (Non-Uniform, 10um Pitch = 1)", fontsize=20, fontweight="bold"
    )
    # ax.set_ylabel("EDP Benefit (10um = 1)", fontsize=20, fontweight="bold")
    # ax.set_ylabel("EDP Benefit \n (Edge, 10um = 1)", fontsize=20, fontweight="bold")
    plt.rc("xtick", labelsize=20)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=20)
    legend_properties = {"weight": "bold"}
    ax.legend(fontsize=16)
    xposition = [1, 8, 256, 25600]
    ax.set_xlim(10, 0.1)

    # ax.set_xticks(3.8*np.arange(7)+len(graph_list)*0.1+0.6*area_range)
    ax.set_xticklabels(["10", "1", "0.1"])
    names = ["DRAM", "5um Pitch", "1um Pitch", "100nm Pitch"]
    colors = ["r", "b", "g", "k"]
    # for i, xc in enumerate(xposition):
    #     plt.axvline(x=xc, label = names[i], color = colors[i], linestyle='--')
    plt.grid(b=True, which="major", axis="y", linewidth=2)
    plt.yscale("log")
    plt.xscale("log")
    fig.tight_layout()

    plt.savefig(
        "figures/area_sweep_nodes" + str(plot) + str(area_range) + "nodes.png",
        bbox_inches="tight",
    )
    plt.show()


def show_memory_capacity_req(graph_list, backprop, names=None, plot="time"):
    """ 
    No. of Monolithic Memory Layers Required by the Workloads 

    Args:
        graph_list (): 
        backprop (): 
        names (, optional): . Defaults to None.
        plot (str, optional): . Defaults to "time".

    Returns:
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    total_mem = []
    for en, graph in enumerate(graph_list):
        if names[en] in ["PageRank"]:
            total_mem.append(11.2)
            continue
        if names[en] in ["SLAM"]:
            total_mem.append(12)
            continue
        if names[en] == "Genomics":
            total_mem.append(6.1)
            continue
        if names[en] == "DLRM":
            factor = 100000
        elif names[en] == "BERT":
            factor = 20000000
        elif names[en] == "RNN":
            factor = 20000
        else:
            factor = 2000000
        weights = 0
        for node in graph.nodes:
            weights += node.weights
        print(weights)
        total_mem.append(weights / factor)
    print(total_mem)
    ax.bar(names, total_mem, width=0.3)
    rects = ax.patches

    # labels = [f"label{i}" for i in range(len(rects))]
    # labels = [
    #     "2.5mm^2",
    #     "2.5mm^2",
    #     "25mm^2",
    #     "25mm^2",
    #     "25mm^2",
    #     "25mm^2",
    #     "250mm^2",
    #     "500mm^2",
    # ]
    # for rect, label in zip(rects, labels):
    #     height = rect.get_height()
    #     ax.text(
    #         rect.get_x() + rect.get_width() / 2,
    #         height,
    #         label,
    #         ha="center",
    #         va="bottom",
    #         fontweight="bold",
    #         # rotation=10,
    #     )

    #     time_list, energy_list, design_list, area = design_runner([graph], backprop, file="illusion.yaml")
    #     # print("design list is ", np.array(design_list)[:,1])
    #     print("============Area is ===========", area)
    #     T1 = np.array(design_list)[:,1]
    #     T2 = [int(x) for x in T1]
    #     # print(T2)
    #     ax.plot(T2, "o-", label=names[en])

    # ax.set_ylabel("No. of Memory Layers Req", fontsize=16, fontweight="bold")
    # plt.rc("xtick", labelsize=12)  # fontsize of the tick labels
    # plt.rc("ytick", labelsize=12)
    # plt.xticks(rotation=80)
    # # ax.set_yticklabels([2,4,6])
    # fig.tight_layout()
    # plt.savefig("figures/memory_cap_req.png", bbox_inches="tight")
    # plt.show()
    return total_mem


def visualize_performance_estimation(mapper, graph, backprop=False):
    """Creates an animated visualization of performance estimation using rectangles
    
    Args:
        mapper: Mapper object containing mapping state
        graph: Input computation graph
        backprop: Whether backpropagation is enabled
        
    Returns:
        List of base64 encoded PNG frames showing the estimation process
    """
    frames = []
    
    # Run mapping to collect event log
    mapper.run_asap(graph)
    events = mapper.event_log
    
    # Create frames for each event
    for event in events:
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Setup the plot area
        ax.set_xlim(-0.5, 2.5)
        ax.set_ylim(-0.5, 1.5)
        ax.axis('off')
        
        # Draw compute unit rectangle
        compute_rect = plt.Rectangle(
            (0, 0), 1, 1, 
            facecolor=f'blue',
            alpha=max(0.1, event['compute_util']),
            edgecolor='black',
            linewidth=2
        )
        ax.add_patch(compute_rect)
        
        # Add compute utilization text
        ax.text(0.5, -0.2, f"Compute\n{event['compute_util']*100:.1f}%", 
                ha='center', va='center')
        
        # Draw memory unit rectangle
        memory_rect = plt.Rectangle(
            (1.5, 0), 1, 1,
            facecolor='green',
            alpha=max(0.1, event['mem_util']),
            edgecolor='black',
            linewidth=2
        )
        ax.add_patch(memory_rect)
        
        # Add memory utilization text
        ax.text(2.0, -0.2, f"Memory\n{event['mem_util']*100:.1f}%",
                ha='center', va='center')
        
        # Draw bandwidth indicator
        if event['bandwidth'] > 0:
            arrow = plt.Arrow(
                1.1, 0.5, 0.3, 0,
                width=0.2,
                color='red',
                alpha=min(1.0, event['bandwidth'] / mapper.mem_read_bw[mapper.mle-1])
            )
            ax.add_patch(arrow)
        
        # Add title with node and phase info
        plt.title(f"Node: {event['node']}\nPhase: {event['phase']}\nCycles: {event['cycles']:.0f}",
                 pad=20, fontsize=12)
        
        # Save frame
        buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format='png', dpi=100)
        plt.close()
        buf.seek(0)
        frames.append(base64.b64encode(buf.getvalue()).decode())
    
    return frames

def generate_system_visualization(perf_results):
    """Generate visualization frames for system-level performance"""
    frames = []
    
    # Create system topology visualization
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot chips and processors
    chip_positions = calculate_node_positions(perf_results)
    
    for pos, chip_result in zip(chip_positions, perf_results['chips']):
        # Draw chip
        rect = plt.Rectangle(
            pos, 1, 1,
            facecolor='blue',
            alpha=chip_result['utilization'],
            label=f"Chip {chip_result['id']}"
        )
        ax.add_patch(rect)
        
        # Add performance metrics
        ax.text(
            pos[0] + 0.5, pos[1] + 0.5,
            f"Perf: {chip_result['performance']:.1f}\nUtil: {chip_result['utilization']*100:.1f}%",
            ha='center', va='center'
        )
    
    # Draw network connections
    for i, latency in enumerate(perf_results['network']['latency_distribution']):
        start = chip_positions[i]
        end = chip_positions[i+1]
        
        # Draw connection with width proportional to bandwidth utilization
        bandwidth_util = perf_results['network']['bandwidth_utilization'][i]
        line = plt.Arrow(
            start[0], start[1],
            end[0] - start[0], end[1] - start[1],
            width=0.1 * bandwidth_util,
            color='red',
            alpha=0.6
        )
        ax.add_patch(line)
        
        # Add latency label
        mid_x = (start[0] + end[0]) / 2
        mid_y = (start[1] + end[1]) / 2
        ax.text(mid_x, mid_y, f"{latency:.1f}ns", ha='center', va='center')
    
    # Add workload distribution info
    for workload, distribution in perf_results['workload_distribution'].items():
        # Add workload allocation visualization
        pass
        
    plt.title("System Performance Visualization")
    
    # Save frame
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    plt.close()
    buf.seek(0)
    frames.append(base64.b64encode(buf.getvalue()).decode())
    
    return frames

def analyze_network_utilization(network, workloads):
    """Analyze network bandwidth utilization for given workloads"""
    try:
        # Calculate data movement requirements
        data_movement = calculate_data_movement(workloads)
        
        # Calculate utilization based on network bandwidth
        bandwidth_gbps = float(network.get('bandwidth', 1))  # Default to 1 GB/s
        utilization = data_movement / (bandwidth_gbps * 1e9)  # Convert GB/s to B/s
        
        return min(1.0, utilization)
    except Exception as e:
        print(f"Error in analyze_network_utilization: {e}")
        return 0.5  # Return default utilization on error

def analyze_network_latency(network, topology):
    """Analyze network latency based on topology"""
    try:
        base_latency = float(network.get('latency', 100))  # Default to 100ns
        
        # Add topology-specific latency factors
        topology_factors = {
            'mesh': 1.2,
            'ring': 1.5,
            'star': 1.0,
            'fully-connected': 1.0
        }
        
        return base_latency * topology_factors.get(str(topology).lower(), 1.0)
    except Exception as e:
        print(f"Error in analyze_network_latency: {e}")
        return 100  # Return default latency on error

def analyze_workload_distribution(workloads, chips, processors):
    """Analyze how workloads are distributed across chips"""
    try:
        distribution = {}
        
        for workload in workloads:
            # Determine optimal chip allocation based on workload type
            if workload in ['ResNet-50', 'BERT', 'GPT-4']:
                # AI workloads - prefer chips with systolic arrays
                allocation = allocate_ai_workload(workload, chips)
            elif workload in ['AES-256', 'SHA-3']:
                # Cryptography workloads - prefer security accelerators
                allocation = allocate_crypto_workload(workload, chips)
            else:
                # General compute - distribute based on available resources
                allocation = allocate_general_workload(workload, chips, processors)
                
            distribution[workload] = allocation
            
        return distribution
    except Exception as e:
        print(f"Error in analyze_workload_distribution: {e}")
        return {}  # Return empty distribution on error

def calculate_data_movement(workloads):
    """Calculate data movement requirements for workloads"""
    data_movement = 0
    for workload in workloads:
        # Add workload-specific data movement calculations
        if workload == "ResNet-50":
            data_movement += 100e9  # 100GB for ResNet-50
        elif workload == "BERT":
            data_movement += 200e9  # 200GB for BERT
        elif workload == "GPT-4":
            data_movement += 500e9  # 500GB for GPT-4
        else:
            data_movement += 50e9  # Default 50GB for other workloads
    return data_movement

def allocate_ai_workload(workload, chips):
    """Allocate AI workload to appropriate chips"""
    allocation = []
    for chip in chips:
        if hasattr(chip, 'mm_compute'):
            # Check for systolic array or MAC units
            if (chip.mm_compute.type1['class'] == 'systolic_array' or 
                chip.mm_compute.type2['class'] == 'mac'):
                allocation.append({
                    'chip_id': chip.name,
                    'utilization': 0.8,
                    'performance_share': 0.4
                })
    return allocation

def allocate_crypto_workload(workload, chips):
    """Allocate cryptography workload to security-optimized chips"""
    allocation = []
    for chip in chips:
        # Check for security features or dedicated crypto units
        if hasattr(chip, 'vector_compute'):
            allocation.append({
                'chip_id': chip.name if hasattr(chip, 'name') else 'unnamed_chip',
                'utilization': 0.6,
                'performance_share': 0.3
            })
    return allocation

def allocate_general_workload(workload, chips, processors):
    """Allocate general compute workload across available resources"""
    allocation = []
    try:
        total_compute = sum(p.cores * p.frequency for p in processors)
        
        for chip in chips:
            compute_share = 0
            if hasattr(chip, 'mm_compute'):
                compute_share = (chip.mm_compute.type1.get('N_PE', 0) * 
                               chip.mm_compute.type1.get('frequency', 1)) / total_compute
            
            allocation.append({
                'chip_id': chip.name if hasattr(chip, 'name') else 'unnamed_chip',
                'utilization': 0.5,
                'performance_share': compute_share
            })
    except Exception as e:
        print(f"Error in allocate_general_workload: {e}")
        # Return default allocation on error
        for i, _ in enumerate(chips):
            allocation.append({
                'chip_id': f'chip_{i}',
                'utilization': 0.5,
                'performance_share': 1.0 / len(chips)
            })
    
    return allocation

def calculate_node_positions(perf_results):
    """Calculate node positions for visualization"""
    positions = []
    num_chips = len(perf_results['chips'])
    
    if perf_results['topology'] == 'mesh':
        # Arrange in grid
        grid_size = math.ceil(math.sqrt(num_chips))
        for i in range(num_chips):
            row = i // grid_size
            col = i % grid_size
            positions.append((col * 2, row * 2))
    elif perf_results['topology'] == 'ring':
        # Arrange in circle
        for i in range(num_chips):
            angle = 2 * math.pi * i / num_chips
            x = math.cos(angle) * 3
            y = math.sin(angle) * 3
            positions.append((x, y))
    else:
        # Default linear arrangement
        for i in range(num_chips):
            positions.append((i * 2, 0))
            
    return positions
