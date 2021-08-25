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
    """[Runs the Input Graph : Optimizes Design and Technology]

    Args:
        graph_set ([type]): [description]
        backprop (bool, optional): [description]. Defaults to False.
        print_stats (bool, optional): [description]. Defaults to False.
        file (str, optional): [description]. Defaults to "default.yaml".
        stats_file (str, optional): [description]. Defaults to "logs/stats.txt".

    Returns:
        [type]: [description]
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
    """[Runs the Input Graph : Optimizes Design Only]

    Args:
        graph_set ([type]): [description]
        backprop (bool, optional): [description]. Defaults to False.
        print_stats (bool, optional): [description]. Defaults to False.
        file (str, optional): [description]. Defaults to "default.yaml".
        stats_file (str, optional): [description]. Defaults to "logs/stats.txt".

    Returns:
        [type]: [description]
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


def perf(
    graph, backprop, print_stats, filename, mapping="nn_dataflow", *args, **kwargs
):
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
    """Plots the Design Parameters Updates in Backward Pass on Running a Given DFG
    Args:
        graph ([type]): [description]
        backprop ([type]): True to Run the Workload in Training
    """
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
    """Plots the Technology Parameters Updates in Backward Pass on Running a Given DFG
    Args:
        graph ([type]): [description]
        backprop ([type]): [description]
    """
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
def s_mem_c_same_arch(
    graph_list, backprop, names=None, plot="time", area_budget=2.5, *args, **kwargs
):
    fig, ax = plt.subplots(figsize=(10, 10))
    for en, graph in enumerate(graph_list):
        time_list = []
        energy_list = []
        for j in range(1, 4000, 4):
            scheduler = Scheduling(hwfile="illusion.yaml")
            generator = Generator()
            scheduler.config["memory"]["level1"]["banks"] = 1
            scheduler.config["memory"]["level1"]["banks"] *= j
            if names[en] == "SSD":
                scheduler.config["memory"]["level0"]["size"] *= 6
            scheduler.complete_config(scheduler.config)
            scheduler.run_asap(graph)
            in_time, in_energy, design, tech, area = generator.save_stats(
                scheduler, backprop, get_backprop_memory(graph.nodes)
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


def s_mem_area_pitch(
    graph_list, backprop, names=None, plot="time", area_budget=2.5, *args, **kwargs
):
    fig, ax = plt.subplots(figsize=(10, 10))
    for en, graph in enumerate(graph_list):
        time_list = []
        energy_list = []
        for j in range(1, 4000, 4):
            scheduler = Scheduling(hwfile="illusion.yaml")
            generator = Generator()
            scheduler.config["memory"]["level1"]["banks"] = 1
            scheduler.config["memory"]["level1"]["banks"] *= j
            if names[en] == "SSD":
                scheduler.config["memory"]["level0"]["size"] *= 6
            scheduler.complete_config(scheduler.config)
            scheduler.run_asap(graph)
            in_time, in_energy, design, tech, area = generator.save_stats(
                scheduler, backprop, get_backprop_memory(graph.nodes)
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
        "figures/pitch_sweep_area" + str(plot) + str(area_budget) + ".png",
        bbox_inches="tight",
    )
    plt.show()


def sweep_area(
    graph_list, backprop, names=None, plot="time", area_range=1, *args, **kwargs
):
    """

    Args:
        graph_list ([type]): [description]
        backprop ([type]): [description]
        names ([type], optional): [description]. Defaults to None.
        plot (str, optional): [description]. Defaults to "time".
        area_range (int, optional): [description]. Defaults to 1.
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
            for m, node in enumerate(node_name):
                pitch_list = []
                pitch_list2 = []
                if names[en] == "PageRank":
                    pitch_list = [
                        10000000,
                        1300000,
                        400000,
                        140000,
                        70000,
                        20000,
                        8000,
                    ]
                elif names[en] == "Genomics":
                    pitch_list = [
                        10000000,
                        900000,
                        400000,
                        200000,
                        100000,
                        70000,
                        50000,
                    ]
                elif names[en] == "SLAM":
                    pitch_list = [
                        10000000,
                        1600000,
                        600000,
                        180000,
                        90000,
                        40000,
                        20000,
                    ]
                else:
                    for i, pitch in enumerate([10, 5, 3, 2, 1, 0.5, 0.1]):
                        percent_time_list = []
                        percent_time_list2 = []
                        for percent in range(5, 95, 10):

                            connectivity_area = (
                                percent * 1000 * 10 ** area_budget / node_density[m]
                            )
                            j = (
                                connectivity_area
                                / (2 * 32 * pitch ** 2)
                                / total_mem[en]
                            )
                            scheduler = Scheduling(hwfile="illusion.yaml")
                            generator = Generator()
                            scheduler.config["mm_compute"]["N_PE"] *= (
                                10 ** area_budget * precision_density_factor[p]
                            )
                            scheduler.config["mm_compute"]["frequency"] *= (
                                node_speed[m] * precision_speed_factor[p]
                            )
                            scheduler.config["mm_compute"][
                                "per_op_energy"
                            ] *= node_energy[m]

                            scheduler.config["memory"]["level0"]["size"] *= (
                                10 ** area_budget * precision_size_factor[p]
                            )
                            scheduler.config["memory"]["level1"]["banks"] = 2
                            scheduler.config["memory"]["level1"]["banks"] *= j
                            scheduler.complete_config(scheduler.config)
                            scheduler.run_asap(graph)
                            (
                                in_time,
                                in_energy,
                                design,
                                tech,
                                area,
                            ) = generator.save_stats(
                                scheduler, backprop, get_backprop_memory(graph.nodes)
                            )
                            new_area = area + connectivity_area
                            percent_time_list2.append(in_time[0])
                            energy_list2.append(in_energy[0])

                            j2 = connectivity_area / (2 * 32 * pitch ** 2)
                            scheduler = Scheduling(hwfile="illusion.yaml")
                            generator = Generator()
                            scheduler.config["mm_compute"]["N_PE"] *= (
                                10 ** area_budget * precision_density_factor[p]
                            )
                            scheduler.config["mm_compute"]["frequency"] *= (
                                node_speed[m] * precision_speed_factor[p]
                            )
                            scheduler.config["mm_compute"][
                                "per_op_energy"
                            ] *= node_energy[m]

                            scheduler.config["memory"]["level0"]["size"] *= (
                                10 ** area_budget * precision_size_factor[p]
                            )
                            scheduler.config["memory"]["level1"]["banks"] = 2
                            scheduler.config["memory"]["level1"]["banks"] *= j2
                            scheduler.complete_config(scheduler.config)
                            scheduler.run_asap(graph)
                            (
                                in_time,
                                in_energy,
                                design,
                                tech,
                                area,
                            ) = generator.save_stats(
                                scheduler, backprop, get_backprop_memory(graph.nodes)
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
    """ No. of Monolithic Memory Layers Required by the Workloads 
    Args:
        graph_list ([type]): [description]
        backprop ([type]): [description]
        names ([type], optional): [description]. Defaults to None.
        plot (str, optional): [description]. Defaults to "time".

    Returns:
        [type]: [description]
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
