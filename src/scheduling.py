import collections
import logging
import pdb

import numpy as np
import yaml
import yamlordereddictloader

from functions import complete_functional_config
from generator import *
from generator import Generator, get_mem_props
from utils.logger import create_logger
from utils.visualizer import *

eff = 0.5


class Scheduling:
    def __init__(self, hwfile="default.yaml", stats_file="logs/stats.txt"):
        base_dir = "configs/"
        self.total_cycles = 0
        self.technology = [1, 1, 40]
        # maybe change this later to peripheral logic node or speed
        #     [wire_cap , sense_amp_time, plogic_node],
        self.logger = create_logger(stats_file=stats_file)
        self.config = self.complete_config(
            yaml.load(open(base_dir + hwfile), Loader=yamlordereddictloader.Loader)
        )


def complete_config(self, config):

    self.logger.debug("Config Statistics : ")

    self.mle = config["memory_levels"]
    self.mem_energy = np.zeros((self.mle))
    self.compute_energy = 0
    self.mem_read_access = np.zeros((self.mle))
    self.mem_write_access = np.zeros((self.mle))
    self.mem_size = np.zeros((self.mle))
    self.mem_util = np.zeros((self.mle))
    self.mem_free = np.zeros((self.mle))
    self.mem_read_bw = np.zeros((self.mle))
    self.mem_write_bw = np.zeros((self.mle))
    self.internal_bandwidth_time = 0
    self.total_cycles = 0
    self.bandwidth_idle_time = 0
    self.compute_idle_time = 0
    self.mem_size_idle_time = 0

    self.force_connectivity = config["force_connectivity"]
    mm_compute = config["mm_compute"]
    vector_compute = config["vector_compute"]

    if mm_compute["class"] == "systolic_array":
        config["mm_compute_per_cycle"] = (
            ((mm_compute["size"]) ** 2) * mm_compute["N_PE"] / (4)
        )
        config["comp_bw"] = (
            mm_compute["size"] * mm_compute["N_PE"] * mm_compute["frequency"] * 2 / 4
        )

        self.logger.debug("MM Compute per cycle : %d", config["mm_compute_per_cycle"])
        self.logger.debug("Compute Bandwidth Required : %d", config["comp_bw"])

    if config["mm_compute"]["class"] == "mac":
        config["mm_compute_per_cycle"] = ((mm_compute["size"])) * mm_compute["N_PE"]
        config["comp_read_bw"] = (
            mm_compute["size"] * mm_compute["N_PE"] * mm_compute["frequency"]
        )

    for i in range(self.mle):
        memory = config["memory"]["level" + str(i)]
        self.mem_read_bw[i] = (
            memory["frequency"]
            * memory["banks"]
            * memory["read_ports"]
            * memory["width"]
        )
        self.mem_write_bw[i] = (
            memory["frequency"]
            * memory["banks"]
            * memory["write_ports"]
            * memory["width"]
        )
        self.mem_size[i] = memory["size"]

        self.logger.debug(
            "Memory at Level %d, Read Bandwidth %d Write Bandwidth %d",
            i,
            self.mem_read_bw[i],
            self.mem_write_bw[i],
        )
    # complete_functional_config
    # complete_performance_config
    # memory
    for i in range(self.mle - 1):
        memory = config["memory"]["level" + str(i)]
        read_energy, write_energy, leakage_power, area = get_mem_props(
            memory["size"], memory["width"], memory["banks"]
        )
        config["memory"]["level" + str(i)]["read_energy"] = str(read_energy)
        config["memory"]["level" + str(i)]["write_energy"] = str(write_energy)
        config["memory"]["level" + str(i)]["leakage_power"] = str(leakage_power)
        config["memory"]["level" + str(i)]["area"] = str(area)
    # compute
    # config["memory"] = mem_space(config["memory"], technology)
    # config["mm_compute"] = comp_space(config["mm_compute"], technology)
    return config


def run_asap(self, graph):

    """
     Check both size, utilization and bandwidths at every node
     What about memory size that can also get exhausted ?
     So if memory size is exhausted, then have to go to a previous level and write there ?
     if any level utilization is exhausted then only the immediate memory required will be kept.
     if the memory is empty in size, but is not bandwidth, it is useless?
     Cannot do prefetching
     Read access of the next node will decrease
     Bandwidth is available but size is not?, can do prefetching, but now the memory fetches have to check, 
     whether to do fetches of the same node or a different node
     Say bandwidth at level0 is sufficient, at level1 is insufficient, then at level1 we have a bottlenecks
     slower so it will take its own time
     Do vector operations in the meantime perhaps ? 

    """

    config = self.config
    # TODO in_edge_mem can change by pooling/batch_norm
    # TODO compute is very much higher than bandwidth time and mem size idle time
    read_bw_req = []
    write_bw_req = []
    read_bw_actual = []
    write_bw_actual = []
    cycles = []
    free_cycles = []
    transferable_checkpointed_edge = []
    all_checkpointed_edge = []
    self.mem_util_log = []
    self.mem_util_full = []
    # Mem Fetch time of the last Nodes
    #     print(self.mem_free[0], self.mem_util[0], self.mem_size[0])

    mem_free = True
    for n, node in enumerate(graph.nodes):
        node.mem_fetch = node.weights
    for n, node in enumerate(graph.nodes):

        # These are last level read/write accesses
        compute_expense, weights = node.get_stats()
        read_access = node.mem_fetch
        write_access = 0
        self.mem_read_access[1] += weights

        assert self.mem_util[0] <= self.mem_size[0]
        self.mem_util[0] += node.in_edge_mem
        node.mem_util = node.out_edge_mem + node.mem_fetch
        # Total Free memory
        for i in range(self.mle - 1):
            self.mem_free[i] = self.mem_size[i] - self.mem_util[i]
        time_compute = compute_expense / config["mm_compute_per_cycle"]
        read_bw_ll = read_access / (time_compute)
        write_bw_ll = write_access / (time_compute)
        step_cycles = time_compute
        read_bw_req.append(read_bw_ll)
        write_bw_req.append(write_bw_ll)
        free_cycles.append(step_cycles)
        n_swaps = 1
        total_mem = 0
        if self.mem_free[0] < node.mem_util:
            mem_free = False
            self.mem_util[0] -= node.in_edge_mem
            self.mem_util[0] -= node.weights - node.mem_fetch
            self.mem_free[0] = self.mem_size[0] - self.mem_util[0]
            total_mem = node.in_edge_mem + node.out_edge_mem + node.weights
            if self.mem_free[0] <= 0:
                print(self.mem_free[0])
            assert self.mem_free[0] > 0, self.mem_util[0]
            n_swaps = total_mem // self.mem_free[0] + 1
            swap_time = max(config["mm_compute"]["size"] * 4, time_compute // n_swaps)
            self.mem_size_idle_time += (
                swap_time * n_swaps
                + ((node.out_edge_mem // n_swaps - 1) * n_swaps)
                // self.mem_read_bw[self.mle - 1]
            )
            self.bandwidth_idle_time += (
                (node.out_edge_mem // n_swaps - 1) * n_swaps
            ) // self.mem_read_bw[self.mle - 1]
            step_cycles += (
                swap_time * n_swaps
                + 2
                * ((node.out_edge_mem // n_swaps - 1) * n_swaps)
                // self.mem_read_bw[self.mle - 1]
            )
            self.mem_read_access[0] += node.mem_util + node.in_edge_mem
            self.mem_write_access[0] += node.mem_util + node.in_edge_mem
            self.mem_write_access[1] += node.out_edge_mem
        else:
            self.mem_util[0] += node.mem_util
            self.mem_free[0] -= node.mem_util
        #         print("2.5",self.mem_free[0], self.mem_util[0], self.mem_size[0])
        self.mem_util_log.append(self.mem_util[0])
        self.mem_read_access[0] += node.weights + node.out_edge_mem
        self.mem_write_access[0] += node.weights + node.out_edge_mem
        assert self.mem_free[0] <= self.mem_size[0]
        # Last level memory fetch takes more time, so that may be a bottleneck
        bandwidth_available = read_bw_ll < self.mem_read_bw[self.mle - 1]

        # If Bandwidth is not available : Cannot Prefetch
        if (bandwidth_available) == False:
            step_cycles += (
                read_bw_ll / self.mem_read_bw[self.mle - 1] - 1
            ) * time_compute
            self.bandwidth_idle_time += (
                read_bw_ll / self.mem_read_bw[self.mle - 1] - 1
            ) * time_compute

        # If memory is not free for the next node and Bandwidth is available : Move nodes back and forth
        # if(total_mem_free[0] == 0 and (bandwidth_available)):
        # for(nodes in checkpointed_nodes):
        # checkpointed but not immediate node

        # Check if memory is free and Bandwidth available : From the Data Dependence Graph, Prefetch new node

        # pdb.set_trace()
        if self.mem_free[0] > 0 and (bandwidth_available):
            if n < len(graph.nodes) - 1:
                if self.mem_free[0] > node.next.mem_fetch:
                    read_access += node.next.mem_fetch
                    if read_access / step_cycles < self.mem_read_bw[self.mle - 1]:
                        self.mem_util[0] += node.next.mem_fetch
                        self.mem_free[0] -= node.next.mem_fetch
                        node.next.mem_fetch = 0
                    else:
                        read_access = self.mem_read_bw[self.mle - 1] * step_cycles
                        self.mem_util[0] += read_access - read_bw_ll * step_cycles
                        self.mem_free[0] -= read_access - read_bw_ll * step_cycles
                        node.next.mem_fetch -= (
                            read_access - read_bw_ll * step_cycles
                        )  # Next node mem fetch gets updated

                else:
                    read_access += self.mem_free[0]
                    if read_access / step_cycles < self.mem_read_bw[self.mle - 1]:
                        node.next.mem_fetch = node.next.mem_fetch - self.mem_free[0]
                        # Next node mem fetch gets updated

                        self.mem_util[0] = self.mem_size[0]
                        self.mem_free[0] = 0
                    else:
                        read_access = self.mem_read_bw[self.mle - 1] * step_cycles
                        self.mem_util[0] += read_access - read_bw_ll * step_cycles
                        self.mem_free[0] -= read_access - read_bw_ll * step_cycles
                        node.next.mem_fetch -= read_access - read_bw_ll * step_cycles
                        # Next node mem fetch gets updated

        #         print("3",self.mem_free[0], self.mem_util[0], self.mem_size[0])
        self.mem_util_full.append(self.mem_util[0])

        # TODO Consider Write bandwidth for a block read memory or Write Bandwidth for endurance purposes
        #         print("4",self.mem_free[0], self.mem_util[0], self.mem_size[0])

        if mem_free:
            self.mem_util[0] -= node.out_edge_mem + node.weights + node.in_edge_mem
        #         print("5",self.mem_free[0], self.mem_util[0], self.mem_size[0])

        self.logger.debug(
            "Node operator %r, Compute Expense %d,   Time Compute %d, Step Cycles %d, Read Accesses %d, Write Accesses %d , No of Swaps %d, Total_mem %d",
            node.operator,
            compute_expense,
            time_compute,
            step_cycles,
            read_access,
            write_access,
            n_swaps,
            total_mem,
        )
        self.total_cycles += step_cycles
        cycles.append(step_cycles)
        read_bw_actual.append(read_access / step_cycles)
        write_bw_actual.append(write_access / step_cycles)
    #         print("actual",read_access / step_cycles, write_access / step_cycles, step_cycles)
    #     print("The total cycles are ", self.total_cycles)
    self.mem_write_access[1] += node.out_edge_mem
    return (
        read_bw_req,
        write_bw_req,
        read_bw_actual,
        write_bw_actual,
        cycles,
        free_cycles,
    )


def run_reuse_leakage(self, graph):
    """
    Energy efficiency mapping with greedy choice between reuse and leakage_power 
    """

    config = self.config
    # TODO in_edge_mem can change by pooling/batch_norm
    # TODO compute is very much higher than bandwidth time and mem size idle time
    read_bw_req = []
    write_bw_req = []
    read_bw_actual = []
    write_bw_actual = []
    cycles = []
    free_cycles = []
    transferable_checkpointed_edge = []
    all_checkpointed_edge = []
    self.mem_util_log = []
    self.mem_util_full = []
    # Mem Fetch time of the last Nodes
    #     print(self.mem_free[0], self.mem_util[0], self.mem_size[0])

    mem_free = True
    for n, node in enumerate(graph.nodes):
        node.mem_fetch = node.weights
    for n, node in enumerate(graph.nodes):

        # These are last level read/write accesses
        compute_expense, weights = node.get_stats()
        """
        mem_reuse, comp_reuse = get_reuse(node)
        time_reuse =         # reusing memory : then time_reuse = time_taken - time_without_reuse
        
        """
        read_access = node.mem_fetch
        write_access = 0
        self.mem_read_access[1] += weights

        assert self.mem_util[0] <= self.mem_size[0]
        self.mem_util[0] += node.in_edge_mem
        node.mem_util = node.out_edge_mem + node.mem_fetch
        # Total Free memory
        for i in range(self.mle - 1):
            self.mem_free[i] = self.mem_size[i] - self.mem_util[i]
        time_compute = compute_expense / config["mm_compute_per_cycle"]
        read_bw_ll = read_access / (time_compute)
        write_bw_ll = write_access / (time_compute)
        step_cycles = time_compute
        read_bw_req.append(read_bw_ll)
        write_bw_req.append(write_bw_ll)
        free_cycles.append(step_cycles)
        n_swaps = 1
        total_mem = 0
        if self.mem_free[0] < node.mem_util:
            mem_free = False
            self.mem_util[0] -= node.in_edge_mem
            self.mem_util[0] -= node.weights - node.mem_fetch
            self.mem_free[0] = self.mem_size[0] - self.mem_util[0]
            total_mem = node.in_edge_mem + node.out_edge_mem + node.weights
            if self.mem_free[0] <= 0:
                print(self.mem_free[0])
            assert self.mem_free[0] > 0, self.mem_util[0]
            n_swaps = total_mem // self.mem_free[0] + 1
            swap_time = max(config["mm_compute"]["size"] * 4, time_compute // n_swaps)
            self.mem_size_idle_time += (
                swap_time * n_swaps
                + ((node.out_edge_mem // n_swaps - 1) * n_swaps)
                // self.mem_read_bw[self.mle - 1]
            )
            self.bandwidth_idle_time += (
                (node.out_edge_mem // n_swaps - 1) * n_swaps
            ) // self.mem_read_bw[self.mle - 1]
            step_cycles += (
                swap_time * n_swaps
                + 2
                * ((node.out_edge_mem // n_swaps - 1) * n_swaps)
                // self.mem_read_bw[self.mle - 1]
            )
            self.mem_read_access[0] += node.mem_util + node.in_edge_mem
            self.mem_write_access[0] += node.mem_util + node.in_edge_mem
            self.mem_write_access[1] += node.out_edge_mem
        else:
            self.mem_util[0] += node.mem_util
            self.mem_free[0] -= node.mem_util
        #         print("2.5",self.mem_free[0], self.mem_util[0], self.mem_size[0])
        self.mem_util_log.append(self.mem_util[0])
        self.mem_read_access[0] += node.weights + node.out_edge_mem
        self.mem_write_access[0] += node.weights + node.out_edge_mem
        assert self.mem_free[0] <= self.mem_size[0]
        # Last level memory fetch takes more time, so that may be a bottleneck
        bandwidth_available = read_bw_ll < self.mem_read_bw[self.mle - 1]

        # If Bandwidth is not available : Cannot Prefetch
        if (bandwidth_available) == False:
            step_cycles += (
                read_bw_ll / self.mem_read_bw[self.mle - 1] - 1
            ) * time_compute
            self.bandwidth_idle_time += (
                read_bw_ll / self.mem_read_bw[self.mle - 1] - 1
            ) * time_compute

        # If memory is not free for the next node and Bandwidth is available : Move nodes back and forth
        # if(total_mem_free[0] == 0 and (bandwidth_available)):
        # for(nodes in checkpointed_nodes):
        # checkpointed but not immediate node

        # Check if memory is free and Bandwidth available : From the Data Dependence Graph, Prefetch new node

        # pdb.set_trace()
        if self.mem_free[0] > 0 and (bandwidth_available):
            if n < len(graph.nodes) - 1:
                if self.mem_free[0] > node.next.mem_fetch:
                    read_access += node.next.mem_fetch
                    if read_access / step_cycles < self.mem_read_bw[self.mle - 1]:
                        self.mem_util[0] += node.next.mem_fetch
                        self.mem_free[0] -= node.next.mem_fetch
                        node.next.mem_fetch = 0
                    else:
                        read_access = self.mem_read_bw[self.mle - 1] * step_cycles
                        self.mem_util[0] += read_access - read_bw_ll * step_cycles
                        self.mem_free[0] -= read_access - read_bw_ll * step_cycles
                        node.next.mem_fetch -= (
                            read_access - read_bw_ll * step_cycles
                        )  # Next node mem fetch gets updated

                else:
                    read_access += self.mem_free[0]
                    if read_access / step_cycles < self.mem_read_bw[self.mle - 1]:
                        node.next.mem_fetch = node.next.mem_fetch - self.mem_free[0]
                        # Next node mem fetch gets updated

                        self.mem_util[0] = self.mem_size[0]
                        self.mem_free[0] = 0
                    else:
                        read_access = self.mem_read_bw[self.mle - 1] * step_cycles
                        self.mem_util[0] += read_access - read_bw_ll * step_cycles
                        self.mem_free[0] -= read_access - read_bw_ll * step_cycles
                        node.next.mem_fetch -= read_access - read_bw_ll * step_cycles
                        # Next node mem fetch gets updated

        #         print("3",self.mem_free[0], self.mem_util[0], self.mem_size[0])
        self.mem_util_full.append(self.mem_util[0])

        # TODO Consider Write bandwidth for a block read memory or Write Bandwidth for endurance purposes
        #         print("4",self.mem_free[0], self.mem_util[0], self.mem_size[0])

        if mem_free:
            self.mem_util[0] -= node.out_edge_mem + node.weights + node.in_edge_mem
        #         print("5",self.mem_free[0], self.mem_util[0], self.mem_size[0])

        self.logger.debug(
            "Node operator %r, Compute Expense %d,   Time Compute %d, Step Cycles %d, Read Accesses %d, Write Accesses %d , No of Swaps %d, Total_mem %d",
            node.operator,
            compute_expense,
            time_compute,
            step_cycles,
            read_access,
            write_access,
            n_swaps,
            total_mem,
        )
        self.total_cycles += step_cycles
        cycles.append(step_cycles)
        read_bw_actual.append(read_access / step_cycles)
        write_bw_actual.append(write_access / step_cycles)
    #         print("actual",read_access / step_cycles, write_access / step_cycles, step_cycles)
    #     print("The total cycles are ", self.total_cycles)
    self.mem_write_access[1] += node.out_edge_mem
    return (
        read_bw_req,
        write_bw_req,
        read_bw_actual,
        write_bw_actual,
        cycles,
        free_cycles,
    )


def run_reuse_full(self, graph):
    """
    Energy efficiency mapping with maximal reuse and power gating of the components
    """
    config = self.config
    # TODO in_edge_mem can change by pooling/batch_norm
    # TODO compute is very much higher than bandwidth time and mem size idle time
    read_bw_req = []
    write_bw_req = []
    read_bw_actual = []
    write_bw_actual = []
    cycles = []
    free_cycles = []
    transferable_checkpointed_edge = []
    all_checkpointed_edge = []
    self.mem_util_log = []
    self.mem_util_full = []
    # Mem Fetch time of the last Nodes
    #     print(self.mem_free[0], self.mem_util[0], self.mem_size[0])

    mem_free = True
    for n, node in enumerate(graph.nodes):
        node.mem_fetch = node.weights
    for n, node in enumerate(graph.nodes):

        # These are last level read/write accesses
        compute_expense, weights = node.get_stats()
        """
        mem_reuse, comp_reuse = get_reuse(node)
        time_reuse = 
        # reusing memory : then time_reuse = time_taken - time_without_reuse
        total_leakage_power = time_reuse *() # sum of leakage power of all
        is_reuse = True
        if(total_leakage_power > mem_reuse*mem_energy):
            is_reuse = False
        """
        read_access = node.mem_fetch
        write_access = 0
        self.mem_read_access[1] += weights

        assert self.mem_util[0] <= self.mem_size[0]
        self.mem_util[0] += node.in_edge_mem
        node.mem_util = node.out_edge_mem + node.mem_fetch
        # Total Free memory
        for i in range(self.mle - 1):
            self.mem_free[i] = self.mem_size[i] - self.mem_util[i]
        time_compute = compute_expense / config["mm_compute_per_cycle"]
        read_bw_ll = read_access / (time_compute)
        write_bw_ll = write_access / (time_compute)
        step_cycles = time_compute
        read_bw_req.append(read_bw_ll)
        write_bw_req.append(write_bw_ll)
        free_cycles.append(step_cycles)
        n_swaps = 1
        total_mem = 0
        if self.mem_free[0] < node.mem_util:
            mem_free = False
            self.mem_util[0] -= node.in_edge_mem
            self.mem_util[0] -= node.weights - node.mem_fetch
            self.mem_free[0] = self.mem_size[0] - self.mem_util[0]
            total_mem = node.in_edge_mem + node.out_edge_mem + node.weights
            if self.mem_free[0] <= 0:
                print(self.mem_free[0])
            assert self.mem_free[0] > 0, self.mem_util[0]
            n_swaps = total_mem // self.mem_free[0] + 1
            swap_time = max(config["mm_compute"]["size"] * 4, time_compute // n_swaps)
            self.mem_size_idle_time += (
                swap_time * n_swaps
                + ((node.out_edge_mem // n_swaps - 1) * n_swaps)
                // self.mem_read_bw[self.mle - 1]
            )
            self.bandwidth_idle_time += (
                (node.out_edge_mem // n_swaps - 1) * n_swaps
            ) // self.mem_read_bw[self.mle - 1]
            step_cycles += (
                swap_time * n_swaps
                + 2
                * ((node.out_edge_mem // n_swaps - 1) * n_swaps)
                // self.mem_read_bw[self.mle - 1]
            )
            self.mem_read_access[0] += node.mem_util + node.in_edge_mem
            self.mem_write_access[0] += node.mem_util + node.in_edge_mem
            self.mem_write_access[1] += node.out_edge_mem
        else:
            self.mem_util[0] += node.mem_util
            self.mem_free[0] -= node.mem_util
        #         print("2.5",self.mem_free[0], self.mem_util[0], self.mem_size[0])
        self.mem_util_log.append(self.mem_util[0])
        self.mem_read_access[0] += node.weights + node.out_edge_mem
        self.mem_write_access[0] += node.weights + node.out_edge_mem
        assert self.mem_free[0] <= self.mem_size[0]
        # Last level memory fetch takes more time, so that may be a bottleneck
        bandwidth_available = read_bw_ll < self.mem_read_bw[self.mle - 1]

        # If Bandwidth is not available : Cannot Prefetch
        if (bandwidth_available) == False:
            step_cycles += (
                read_bw_ll / self.mem_read_bw[self.mle - 1] - 1
            ) * time_compute
            self.bandwidth_idle_time += (
                read_bw_ll / self.mem_read_bw[self.mle - 1] - 1
            ) * time_compute

        # If memory is not free for the next node and Bandwidth is available : Move nodes back and forth
        # if(total_mem_free[0] == 0 and (bandwidth_available)):
        # for(nodes in checkpointed_nodes):
        # checkpointed but not immediate node

        # Check if memory is free and Bandwidth available : From the Data Dependence Graph, Prefetch new node

        # pdb.set_trace()
        if self.mem_free[0] > 0 and (bandwidth_available):
            if n < len(graph.nodes) - 1:
                if self.mem_free[0] > node.next.mem_fetch:
                    read_access += node.next.mem_fetch
                    if read_access / step_cycles < self.mem_read_bw[self.mle - 1]:
                        self.mem_util[0] += node.next.mem_fetch
                        self.mem_free[0] -= node.next.mem_fetch
                        node.next.mem_fetch = 0
                    else:
                        read_access = self.mem_read_bw[self.mle - 1] * step_cycles
                        self.mem_util[0] += read_access - read_bw_ll * step_cycles
                        self.mem_free[0] -= read_access - read_bw_ll * step_cycles
                        node.next.mem_fetch -= (
                            read_access - read_bw_ll * step_cycles
                        )  # Next node mem fetch gets updated

                else:
                    read_access += self.mem_free[0]
                    if read_access / step_cycles < self.mem_read_bw[self.mle - 1]:
                        node.next.mem_fetch = node.next.mem_fetch - self.mem_free[0]
                        # Next node mem fetch gets updated

                        self.mem_util[0] = self.mem_size[0]
                        self.mem_free[0] = 0
                    else:
                        read_access = self.mem_read_bw[self.mle - 1] * step_cycles
                        self.mem_util[0] += read_access - read_bw_ll * step_cycles
                        self.mem_free[0] -= read_access - read_bw_ll * step_cycles
                        node.next.mem_fetch -= read_access - read_bw_ll * step_cycles
                        # Next node mem fetch gets updated

        #         print("3",self.mem_free[0], self.mem_util[0], self.mem_size[0])
        self.mem_util_full.append(self.mem_util[0])

        # TODO Consider Write bandwidth for a block read memory or Write Bandwidth for endurance purposes
        #         print("4",self.mem_free[0], self.mem_util[0], self.mem_size[0])

        if mem_free:
            self.mem_util[0] -= node.out_edge_mem + node.weights + node.in_edge_mem
        #         print("5",self.mem_free[0], self.mem_util[0], self.mem_size[0])

        self.logger.debug(
            "Node operator %r, Compute Expense %d,   Time Compute %d, Step Cycles %d, Read Accesses %d, Write Accesses %d , No of Swaps %d, Total_mem %d",
            node.operator,
            compute_expense,
            time_compute,
            step_cycles,
            read_access,
            write_access,
            n_swaps,
            total_mem,
        )
        self.total_cycles += step_cycles
        cycles.append(step_cycles)
        read_bw_actual.append(read_access / step_cycles)
        write_bw_actual.append(write_access / step_cycles)
    #         print("actual",read_access / step_cycles, write_access / step_cycles, step_cycles)
    #     print("The total cycles are ", self.total_cycles)
    self.mem_write_access[1] += node.out_edge_mem
    return (
        read_bw_req,
        write_bw_req,
        read_bw_actual,
        write_bw_actual,
        cycles,
        free_cycles,
    )


def run_nn_dataflow(self, graph):
    config = self.config
    # TODO in_edge_mem can change by pooling/batch_norm
    # TODO compute is very much higher than bandwidth time and mem size idle time
    read_bw_req = []
    write_bw_req = []
    read_bw_actual = []
    write_bw_actual = []
    cycles = []
    free_cycles = []
    transferable_checkpointed_edge = []
    all_checkpointed_edge = []
    self.mem_util_log = []
    self.mem_util_full = []
    # Mem Fetch time of the last Nodes
    #     print(self.mem_free[0], self.mem_util[0], self.mem_size[0])

    mem_free = True
    for n, node in enumerate(graph.nodes):
        node.mem_fetch = node.weights
    for n, node in enumerate(graph.nodes):

        # These are last level read/write accesses
        compute_expense, weights = node.get_stats()
        """
        mem_reuse, comp_reuse = get_reuse(node)
        time_reuse = 
        # reusing memory : then time_reuse = time_taken - time_without_reuse
        total_leakage_power = time_reuse *() # sum of leakage power of all
        is_reuse = True
        if(total_leakage_power > mem_reuse*mem_energy):
            is_reuse = False
        """
        read_access = node.mem_fetch
        write_access = 0
        self.mem_read_access[1] += weights

        assert self.mem_util[0] <= self.mem_size[0]
        self.mem_util[0] += node.in_edge_mem
        node.mem_util = node.out_edge_mem + node.mem_fetch
        # Total Free memory
        for i in range(self.mle - 1):
            self.mem_free[i] = self.mem_size[i] - self.mem_util[i]
        time_compute = compute_expense / config["mm_compute_per_cycle"]
        read_bw_ll = read_access / (time_compute)
        write_bw_ll = write_access / (time_compute)
        step_cycles = time_compute
        read_bw_req.append(read_bw_ll)
        write_bw_req.append(write_bw_ll)
        free_cycles.append(step_cycles)
        n_swaps = 1
        total_mem = 0
        if self.mem_free[0] < node.mem_util:
            mem_free = False
            self.mem_util[0] -= node.in_edge_mem
            self.mem_util[0] -= node.weights - node.mem_fetch
            self.mem_free[0] = self.mem_size[0] - self.mem_util[0]
            total_mem = node.in_edge_mem + node.out_edge_mem + node.weights
            if self.mem_free[0] <= 0:
                print(self.mem_free[0])
            assert self.mem_free[0] > 0, self.mem_util[0]
            n_swaps = total_mem // self.mem_free[0] + 1
            swap_time = max(config["mm_compute"]["size"] * 4, time_compute // n_swaps)
            self.mem_size_idle_time += (
                swap_time * n_swaps
                + ((node.out_edge_mem // n_swaps - 1) * n_swaps)
                // self.mem_read_bw[self.mle - 1]
            )
            self.bandwidth_idle_time += (
                (node.out_edge_mem // n_swaps - 1) * n_swaps
            ) // self.mem_read_bw[self.mle - 1]
            step_cycles += (
                swap_time * n_swaps
                + 2
                * ((node.out_edge_mem // n_swaps - 1) * n_swaps)
                // self.mem_read_bw[self.mle - 1]
            )
            self.mem_read_access[0] += node.mem_util + node.in_edge_mem
            self.mem_write_access[0] += node.mem_util + node.in_edge_mem
            self.mem_write_access[1] += node.out_edge_mem
        else:
            self.mem_util[0] += node.mem_util
            self.mem_free[0] -= node.mem_util
        #         print("2.5",self.mem_free[0], self.mem_util[0], self.mem_size[0])
        self.mem_util_log.append(self.mem_util[0])
        self.mem_read_access[0] += node.weights + node.out_edge_mem
        self.mem_write_access[0] += node.weights + node.out_edge_mem
        assert self.mem_free[0] <= self.mem_size[0]
        # Last level memory fetch takes more time, so that may be a bottleneck
        step_cycles += read_access / self.mem_read_bw[self.mle - 1] + 1
        self.bandwidth_idle_time += read_access / self.mem_read_bw[self.mle - 1] + 1

        self.mem_util_full.append(self.mem_util[0])

        # TODO Consider Write bandwidth for a block read memory or Write Bandwidth for endurance purposes
        #         print("4",self.mem_free[0], self.mem_util[0], self.mem_size[0])

        if mem_free:
            self.mem_util[0] -= node.out_edge_mem + node.weights + node.in_edge_mem
        #         print("5",self.mem_free[0], self.mem_util[0], self.mem_size[0])

        self.logger.debug(
            "Node operator %r, Compute Expense %d,   Time Compute %d, Step Cycles %d, Read Accesses %d, Write Accesses %d , No of Swaps %d, Total_mem %d",
            node.operator,
            compute_expense,
            time_compute,
            step_cycles,
            read_access,
            write_access,
            n_swaps,
            total_mem,
        )
        self.total_cycles += step_cycles
        cycles.append(step_cycles)
        read_bw_actual.append(read_access / step_cycles)
        write_bw_actual.append(write_access / step_cycles)
        #         print("actual",read_access / step_cycles, write_access / step_cycles, step_cycles)
        #     print("The total cycles are ", self.total_cycles)
        self.mem_write_access[1] += node.out_edge_mem
    # total_fetch = 0
    # for i, node in enumerate(graph.nodes):
    #     total_fetch += (node.in_edge_mem) // 2
    # step_cycles += total_fetch / self.mem_read_bw[self.mle - 1] + 1
    # self.bandwidth_idle_time += total_fetch / self.mem_read_bw[self.mle - 1] + 1
    return (
        read_bw_req,
        write_bw_req,
        read_bw_actual,
        write_bw_actual,
        cycles,
        free_cycles,
    )


def get_reuse(node):
    # for node.type in conv2d
    #
    pass


Scheduling.complete_config = complete_config
Scheduling.run_asap = run_asap
Scheduling.run_reuse_full = run_reuse_full
Scheduling.run_reuse_leakage = run_reuse_leakage
Scheduling.run_nn_dataflow = run_nn_dataflow
