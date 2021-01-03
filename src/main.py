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
            config = generator.backward_pass(scheduler)
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
            config = generator.backward_pass(scheduler)
            generator.writeconfig(config, str(i) + "hw.yaml")
            scheduler.complete_config(config)
            time_list.append(time)
            energy_list.append(energy)
            design_list.append(design)
            i += 1
        print(in_time[0] / time[0], in_energy[0] / energy[0], in_area / area)

    return time, energy, area
    # return time_list, energy_list, design_list


def run_single(graph, backprop, print_stats, filename):
    scheduler = Scheduling(hwfile=filename)
    scheduler.run_asap(graph)
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
        config = generator.backward_pass(scheduler)
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
        config = generator.backward_pass(scheduler)
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
def s_mem_c_same_arch(graph, backprop):
    fig, ax = plt.subplots(figsize=(10, 10))
    for en, graph in enumerate(graph_list):
        time_list = []
        energy_list = []
        for j in range(1, 100):
            scheduler = Scheduling()
            generator = Generator()
            backprop = True
            scheduler.config["memory"]["level1"]["banks"] = 2
            scheduler.config["memory"]["level1"]["banks"] *= j
            scheduler.complete_config(scheduler.config)
            in_time, in_energy, design, tech, area = generator.save_stats(
                scheduler, backprop, get_backprop_memory(graph.nodes)
            )
            time_list.append(in_time[0])
            energy_list.append(in_energy[0])
        ax.plot(energy_list, "o-", label=name[en])
        print(energy_list[0] / energy_list[98])
        print(time_list[0] / time_list[98])
        ax.plot(time_list, "o-", label=name)

    ax.set_xlabel("Memory Connectivity", fontsize=20, fontweight="bold")
    ax.set_ylabel("Energy Consumption", fontsize=20, fontweight="bold")
    plt.rc("xtick", labelsize=20)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=20)
    ax.legend(fontsize=20)
    plt.yscale("log")
    fig.tight_layout()
    plt.savefig("figures/connectivity_sweep_energy.png", bbox_inches="tight")
    plt.show()


# Allow Architecture to Change when Sweeping Memory Connectivity
def s_mem_c_diff_arch(graph, backprop):
    fig, ax = plt.subplots(figsize=(10, 10))
    for en, graph in enumerate(graph_list):
        time_list = []
        energy_list = []
        for j in range(1, 100):
            scheduler = Scheduling()
            generator = Generator()
            backprop = True
            scheduler.config["memory"]["level1"]["banks"] = 2
            scheduler.config["memory"]["level1"]["banks"] *= j
            scheduler.complete_config(scheduler.config)
            scheduler.run_asap(graph)
            in_time, in_energy, design, tech = generator.save_stats(
                scheduler, backprop, get_backprop_memory(graph.nodes)
            )
            time_list.append(in_time[0])
            energy_list.append(in_energy[0])
        ax.plot(energy_list, "o-", label=name[en])
        print(energy_list[0] / energy_list[98])
        print(time_list[0] / time_list[98])
        ax.plot(time_list, "o-", label=name)

    ax.set_xlabel("Memory Connectivity", fontsize=20, fontweight="bold")
    ax.set_ylabel("Energy Consumption", fontsize=20, fontweight="bold")
    plt.rc("xtick", labelsize=20)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=20)
    ax.legend(fontsize=20)
    plt.yscale("log")
    fig.tight_layout()
    plt.savefig("figures/connectivity_sweep_energy.png", bbox_inches="tight")
    plt.show()


# Change Memory Connectivity and Memory Size in Conjuction see how those two are correlated
def s_size_c_joint(graph, backprop):
    fig, ax = plt.subplots(figsize=(10, 10))
    for en, graph in enumerate(graph_list):
        time_list = []
        energy_list = []
        for j in range(1, 100):
            scheduler = Scheduling()
            generator = Generator()
            backprop = True
            scheduler.config["memory"]["level1"]["banks"] = 2
            scheduler.config["memory"]["level1"]["banks"] *= j
            scheduler.complete_config(scheduler.config)
            in_time, in_energy, design, tech = generator.save_stats(
                scheduler, backprop, get_backprop_memory(graph.nodes)
            )
            time_list.append(in_time[0])
            energy_list.append(in_energy[0])
        ax.plot(energy_list, "o-", label=name[en])
        print(energy_list[0] / energy_list[98])
        print(time_list[0] / time_list[98])
        ax.plot(time_list, "o-", label=name)
    ax.set_xlabel("Memory Connectivity", fontsize=20, fontweight="bold")
    ax.set_ylabel("Energy Consumption", fontsize=20, fontweight="bold")
    plt.rc("xtick", labelsize=20)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=20)
    ax.legend(fontsize=20)
    plt.yscale("log")
    fig.tight_layout()
    plt.savefig("figures/connectivity_sweep_energy.png", bbox_inches="tight")
    plt.show()
