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
from generator import Generator, get_mem_props
from ir.handlers import handlers
from ir.trace import get_backprop_memory, trace
from scheduling import Scheduling
from utils.visualizer import *
from utils.visualizer import (
    bandwidth_bar_graph,
    cycles_bar_graph,
    mem_util_bar_graph,
    plot_gradients,
)

####################################


def run_mapping(scheduler, mapping, graph):
    if mapping == "asap":
        scheduler.run_asap(graph)
    elif mapping == "nn_dataflow":
        scheduler.run_nn_dataflow(graph)
    elif mapping == "reuse_full":
        scheduler.run_reuse_full(graph)
    elif mapping == "reuse_leakage":
        scheduler.run_reuse_leakage(graph)


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
        scheduler = Scheduling(stats_file=stats_file)
        scheduler.run_asap(graph)
        in_time, in_energy, design, tech, in_area = generator.save_stats(
            scheduler, backprop, get_backprop_memory(graph.nodes), print_stats
        )
        scheduler = Scheduling(stats_file=stats_file)
        ("======Optimizing Design and Connectivity=========")
        i = 0
        while True:
            _, _, _, _, cycles, free_cycles = scheduler.run_asap(graph)
            time, energy, design, tech, area = generator.save_stats(
                scheduler, backprop, get_backprop_memory(graph.nodes), _stats
            )
            if (
                scheduler.bandwidth_idle_time < 0.1 * scheduler.total_cycles
                or scheduler.force_connectivity
            ) and scheduler.mem_size_idle_time < 0.1 * scheduler.total_cycles:
                break
            # (area / in_area)
            config = generator.backward_pass_design(scheduler)
            generator.writeconfig(config, str(i) + "hw.yaml")
            scheduler.complete_config(config)
            i += 1

        (in_time[0] / time[0], in_energy[0] / energy[0], in_area[0] / area[0])
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
        scheduler = Scheduling(hwfile=file, stats_file=stats_file)
        scheduler.run_asap(graph)
        in_time, in_energy, in_design, in_tech, in_area = generator.save_stats(
            scheduler, backprop, get_backprop_memory(graph.nodes), print_stats
        )
        scheduler = Scheduling(hwfile=file, stats_file=stats_file)
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
        print(
            "Faster : ",
            in_time[0] / time[0],
            "Energy Improvement : ",
            in_energy[0] / energy[0],
            "Area Budget : ",
            in_area / area,
        )

    # return time, energy, area
    return time_list, energy_list, design_list, area


def perf(graph, backprop, print_stats, filename, mapping="nn_dataflow", *args, **kwargs):
    scheduler = Scheduling(hwfile=filename)
    if mapping == "asap":
        scheduler.run_asap(graph)
    elif mapping == "nn_dataflow":
        scheduler.run_nn_dataflow(graph)
    elif mapping == "reuse_full":
        scheduler.run_reuse_full(graph)
    elif mapping == "reuse_leakage":
        scheduler.run_reuse_leakage(graph)
    generator = Generator()
    in_time, in_energy, design, tech, area = generator.save_stats(
        scheduler, backprop, get_backprop_memory(graph.nodes), print_stats
    )
    return in_time, in_energy, area


def all_design_updates(graph, backprop):
    fig, ax = plt.subplots(figsize=(10, 10))
    # ax2 = ax.twinx()
    base_dir = "figures/"
    design_list = []
    design_names = []
    time_list = []
    energy_list = []
    scheduler = Scheduling()
    scheduler.run_asap(graph)
    generator = Generator()
    print_stats = True
    in_time, in_energy, in_design, in_tech, in_area = generator.save_stats(
        scheduler, backprop, get_backprop_memory(graph.nodes), print_stats
    )
    for i in range(num_iterations):
        config = generator.backward_pass_design(scheduler)
        generator.writeconfig(config, str(i) + "hw.yaml")
        scheduler.complete_config(config)
        _, _, _, _, cycles, free_cycles = scheduler.run_asap(graph)
        time, energy, design, tech, area = generator.save_stats(
            scheduler, backprop, get_backprop_memory(graph.nodes), print_stats
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
    fig, ax = plt.subplots(figsize=(10, 10))
    # ax2 = ax.twinx()
    tech_names = []
    tech_list = []
    time_list = []
    energy_list = []
    base_dir = "figures/"
    scheduler = Scheduling()
    scheduler.run_asap(graph)
    generator = Generator()
    print_stats = True
    in_time, in_energy, in_design, in_tech, in_area = generator.save_stats(
        scheduler, backprop, get_backprop_memory(graph.nodes), print_stats
    )
    for i in range(num_iterations):
        config = generator.backward_pass_design(scheduler)
        generator.writeconfig(config, str(i) + "hw.yaml")
        scheduler.complete_config(config)
        _, _, _, _, cycles, free_cycles = scheduler.run_asap(graph)
        time, energy, design, tech, area = generator.save_stats(
            scheduler, backprop, get_backprop_memory(graph.nodes), print_stats
        )
    for i in range(10):
        config = generator.backward_pass_tech(scheduler, "time")
        generator.writeconfig(config, str(i) + "hw.yaml")
        scheduler.complete_config(config)
        _, _, _, _, cycles, free_cycles = scheduler.run_asap(graph)
        time, energy, design, tech, area = generator.save_stats(
            scheduler, backprop, get_backprop_memory(graph.nodes), print_stats
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


# Fix Everything in Architecture and Just Sweep Memory Connectivity
def s_mem_c_same_arch(graph_list, backprop, names=None, plot="time", area_budget=2.5, *args, **kwargs):
    fig, ax = plt.subplots(figsize=(10, 10))
    for en, graph in enumerate(graph_list):
        time_list = []
        energy_list = []
        for j in range(1, 4000, 4):
            scheduler = Scheduling(hwfile="illusion.yaml")
            generator = Generator()
            scheduler.config["memory"]["level1"]["banks"] = 1
            scheduler.config["memory"]["level1"]["banks"] *= j
            if names[en]=="SSD":
                scheduler.config["memory"]["level0"]["size"] *= 6
            scheduler.complete_config(scheduler.config)
            scheduler.run_asap(graph)
            in_time, in_energy, design, tech, area = generator.save_stats(
                scheduler, backprop, get_backprop_memory(graph.nodes)
            )
            time_list.append(in_time[0])
            energy_list.append(in_energy[0])
        if plot =="time":
            if names[en]=="DLRM":
                time_list = np.array(time_list)*50   
            ax.plot(np.arange(2,8000,8),time_list, "o-", label=names[en])
        elif plot =="energy":
            ax.plot(energy_list, "o-", label=names[en])
        else:
            # plot edp
            ax.plot([x*energy_list[enum] for enum, x in enumerate(time_list)], "o-", label=names[en])
    # ax.plot()
    ax.set_xlabel("Memory Connectivity", fontsize=20, fontweight="bold")
    ax.set_ylabel("EDP", fontsize=20, fontweight="bold")
    plt.rc("xtick", labelsize=20)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=20)
    xposition = [1,8, 256, 25600]
    names = ["DRAM", "5um Pitch", "1um Pitch", "100nm Pitch"]
    colors = ["r", 'b', 'g', 'k']
    for i, xc in enumerate(xposition):
        plt.axvline(x=xc, label = names[i], color = colors[i], linestyle='--')
    ax.legend(fontsize=20)
    plt.yscale("log")
    plt.xscale("log")
    fig.tight_layout()
    plt.savefig("figures/connectivity_sweep_area"+str(plot)+str(area_budget)+".png", bbox_inches="tight")
    plt.show()

def s_mem_area_pitch(graph_list, backprop, names=None, plot="time", area_budget=2.5, *args, **kwargs):
    fig, ax = plt.subplots(figsize=(10, 10))
    for en, graph in enumerate(graph_list):
        time_list = []
        energy_list = []
        for j in range(1, 4000, 4):
            scheduler = Scheduling(hwfile="illusion.yaml")
            generator = Generator()
            scheduler.config["memory"]["level1"]["banks"] = 1
            scheduler.config["memory"]["level1"]["banks"] *= j
            if names[en]=="SSD":
                scheduler.config["memory"]["level0"]["size"] *= 6
            scheduler.complete_config(scheduler.config)
            scheduler.run_asap(graph)
            in_time, in_energy, design, tech, area = generator.save_stats(
                scheduler, backprop, get_backprop_memory(graph.nodes)
            )
            time_list.append(in_time[0])
            energy_list.append(in_energy[0])
        if plot =="time":
            if names[en]=="DLRM":
                time_list = np.array(time_list)*50   
            ax.plot(np.arange(2,8000,8),time_list, "o-", label=names[en])
        elif plot =="energy":
            ax.plot(energy_list, "o-", label=names[en])
        else:
            # plot edp
            ax.plot([x*energy_list[enum] for enum, x in enumerate(time_list)], "o-", label=names[en])
    # ax.plot()
    ax.set_xlabel("Memory Connectivity", fontsize=20, fontweight="bold")
    ax.set_ylabel("EDP", fontsize=20, fontweight="bold")
    plt.rc("xtick", labelsize=20)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=20)
    xposition = [1,8, 256, 25600]
    names = ["DRAM", "5um Pitch", "1um Pitch", "100nm Pitch"]
    colors = ["r", 'b', 'g', 'k']
    for i, xc in enumerate(xposition):
        plt.axvline(x=xc, label = names[i], color = colors[i], linestyle='--')
    ax.legend(fontsize=20)
    plt.yscale("log")
    plt.xscale("log")
    fig.tight_layout()
    plt.savefig("figures/pitch_sweep_area"+str(plot)+str(area_budget)+".png", bbox_inches="tight")
    plt.show()


def sweep_area(graph_list, backprop, names=None, plot="time", area_budget=2.5, *args, **kwargs):
    fig, ax = plt.subplots(figsize=(12, 12))
    
    area_names = ["Edge", "Mid", "Cloud", "Giant", "Wafer-Scale"]
    markers_plot = ['o', 'x', '>', ">", '.', '--']
    colors = ['r', 'b', 'g', 'k', 'c', 'm','y']
    density_factor = [16, 64]
    speed_factor = [4,8]
    power_factor = [20, 40]
    size_factor = [4,8]

    for en, graph in enumerate(graph_list):
            # if(names[en]=="SSD"):
            #     print_stats = True
            # else:
            print_stats = False
            for area_budget in range(1):
                energy_list = []
                # connectivity = 2*j*32
                for m, precision in enumerate([4,8]):
                    pitch_list = []
                    for i,pitch in enumerate([10,5,3,2,1,0.5,0.1]):
                            percent_time_list = []
                            for percent in range(5,95,10):
                                connectivity_area = percent*1000*10**area_budget/16
                                j = connectivity_area/(2*32*pitch**2)
                                scheduler = Scheduling(hwfile="illusion.yaml")
                                generator = Generator()
                                scheduler.config["mm_compute"]["N_PE"] *= density_factor[m]*10**area_budget
                                scheduler.config["mm_compute"]["frequency"] *= speed_factor[m]
                                scheduler.config["mm_compute"]["per_op_energy"] /= power_factor[m]

                                scheduler.config["memory"]["level0"]["size"] *= size_factor[m]*10**area_budget
                                scheduler.config["memory"]["level1"]["banks"] = 2
                                scheduler.config["memory"]["level1"]["banks"] *= j
                                scheduler.complete_config(scheduler.config)
                                scheduler.run_asap(graph)
                                in_time, in_energy, design, tech, area = generator.save_stats(
                                    scheduler, backprop, get_backprop_memory(graph.nodes), print_stats
                                )
                                new_area = area/17 + connectivity_area
                                percent_time_list.append(in_time[0])
                                energy_list.append(in_energy[0])
                            pitch_list.append(min(percent_time_list))
                        # np.arange(2,8000,8)
                    if plot =="time":   
                        ax.plot(['10','5','3','2','1','0.5','0.1'],pitch_list, color = colors[en],marker=markers_plot[area_budget], label=area_names[area_budget]+names[en] + "precision : "+ str(precision))
                        # ax.plot(['10','5','3','2','1','0.5','0.1'],pitch_list, color = colors[en],marker=markers_plot[area_budget], label=area_names[area_budget]+names[en] + " fp32")
                    elif plot =="energy":
                        ax.plot(energy_list, "o-", label=names[en])
                    else:
                        # plot edp
                        ax.plot([x*energy_list[enum] for enum, x in enumerate(time_list)], "o-", label=names[en])
        
    ax.set_xlabel("Pitch", fontsize=20, fontweight="bold")
    ax.set_ylabel("EDP", fontsize=20, fontweight="bold")
    plt.rc("xtick", labelsize=20)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=20)
    xposition = [1,8, 256, 25600]
    # ax.set_ylim(1e5,1e7)
    names = ["DRAM", "5um Pitch", "1um Pitch", "100nm Pitch"]
    colors = ["r", 'b', 'g', 'k']
    # for i, xc in enumerate(xposition):
    #     plt.axvline(x=xc, label = names[i], color = colors[i], linestyle='--')
    plt.yscale("log")
    fig.tight_layout()
    ax.legend(fontsize=12)
    plt.savefig("figures/area_sweep"+str(plot)+str(area_budget)+".png", bbox_inches="tight")
    plt.show()

def show_memory_capacity_req(graph_list, backprop, names=None, plot="time", area_budget=2.5):
    fig, ax = plt.subplots(figsize=(10, 10))
    total_mem = []
    for en, graph in enumerate(graph_list):
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
            weights+= node.weights
        print(weights)
        total_mem.append(weights/factor)
    print(total_mem)
    ax.bar(names, total_mem, width = 0.3)

    #     time_list, energy_list, design_list, area = design_runner([graph], backprop, file="illusion.yaml")
    #     # print("design list is ", np.array(design_list)[:,1])
    #     print("============Area is ===========", area)
    #     T1 = np.array(design_list)[:,1]
    #     T2 = [int(x) for x in T1]
    #     # print(T2)
    #     ax.plot(T2, "o-", label=names[en])
   
    ax.set_ylabel("Memory Capacity (No. of Layers Req)", fontsize=20, fontweight="bold")
    plt.rc("xtick", labelsize=20)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=20)
    # ax.set_yticklabels([2,4,6])
    ax.legend(fontsize=20)
    fig.tight_layout()
    plt.savefig("figures/memory_cap_req"+str(plot)+str(area_budget)+".png", bbox_inches="tight")
    plt.show()


# Change Memory Connectivity and Memory Size in Conjuction see how those two are correlated
def s_size_c_joint(graph, backprop):
    fig, ax = plt.subplots(figsize=(10, 10))
    for en, graph in enumerate(graph):
        time_list = []
        energy_list = []
        for j in range(1, 100):
            scheduler = Scheduling()
            generator = Generator()
            scheduler.config["memory"]["level1"]["banks"] = 2
            scheduler.config["memory"]["level1"]["banks"] *= j
            scheduler.complete_config(scheduler.config)
            scheduler.run_asap(graph)
            # scheduler.config["memory"]["level0"]["size"] *= 2
            in_time, in_energy, design, tech = generator.save_stats(
                scheduler, backprop, get_backprop_memory(graph.nodes)
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
