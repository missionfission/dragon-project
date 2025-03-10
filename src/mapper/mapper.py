import collections
import logging
import pdb

import numpy as np
import yaml
import yamlordereddictloader

from generator import *
from generator import Generator, get_mem_props
from synthesis import ai_utils
from utils.logger import create_logger
from utils.visualizer import *

eff = 0.5


class Mapper:
    def __init__(self, hwfile="default.yaml", stats_file="logs/stats.txt"):
        """Mapper Class that Provides Interface for Different Mappings on the Architecture 

        Args:
            hwfile (str, optional): . Defaults to "default.yaml".
            stats_file (str, optional): . Defaults to "logs/stats.txt".
        """
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
        """
        1. Completes the Config for Hardware Description by using Technology/Design Functions
        2. Adds State Variables as Args that are Tracked for ASAP Mapping

        Args:
            config (): 

        Returns:
            : 
        """

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

        if mm_compute["type1"]["class_type"] == "systolic_array":
            config["mm_compute_per_cycle"] = (
                ((mm_compute["type1"]["size"]) ** 2) * mm_compute["type1"]["N_PE"] / (4)
            )
            config["comp_bw"] = (
                mm_compute["type1"]["size"]
                * mm_compute["type1"]["N_PE"]
                * mm_compute["type1"]["frequency"]
                * 2
                / 4
            )

            self.logger.debug(
                "MM Compute per cycle : %d", config["mm_compute_per_cycle"]
            )
            self.logger.debug("Compute Bandwidth Required : %d", config["comp_bw"])

        if config["mm_compute"]["type1"]["class_type"] == "mac":
            config["mm_compute_per_cycle"] = (mm_compute["type1"]["size"]) * mm_compute["type1"]["N_PE"]
            config["comp_read_bw"] = (
                mm_compute["type1"]["size"] * mm_compute["type1"]["N_PE"] * mm_compute["type1"]["frequency"]
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

    def dataflow_solver_wrapper(self, node):
        """
        Wrapper around dataflow solver to calculate compute time for a node.
        This implements a simplified version that estimates based on compute requirements.

        Args:
            node: The node to calculate compute time for

        Returns:
            float: Estimated compute time in cycles
        """
        compute_expense, _ = node.get_stats()
        
        # Get compute capabilities from config
        compute_per_cycle = self.config["mm_compute_per_cycle"]
        
        # Basic computation time
        time_compute = compute_expense / compute_per_cycle
        
        # Add overhead for dataflow scheduling
        scheduling_overhead = 1.1  # 10% overhead for dataflow scheduling
        
        return time_compute * scheduling_overhead

    def run_asap(self, graph):
        """
        Runs the Graph on the Hardware ASAP Mapped with FX graph edge awareness

        We following a Dynamic State Variable Execution Mapping:
        1. Memory Size and Maximum Allowed Bandwidth are taken from Hardware Config
        2. Current Memory Size and Memory Utilization are calculated by Mapping the Nodes Serially
        3. Uses FX graph edges for better prefetching decisions
        """
        config = self.config
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

        # Add performance tracking arrays
        self.compute_util_log = []  # Track compute utilization
        self.memory_bandwidth_log = []  # Track memory bandwidth usage
        self.cycle_count_log = []  # Track cycle counts
        self.event_log = []  # Track events for animation

        mem_free = True
        print(graph)
        # Initialize memory fetch requirements
        for node in graph.nodes:
            node.mem_fetch = node.weights
            
        # Process nodes in topological order
        for n, node in enumerate(graph.nodes):
            # Log initial state
            event = {
                'step': n,
                'node': node.operator,
                'phase': 'start',
                'compute_util': 0,
                'mem_util': self.mem_util[0] / self.mem_size[0],
                'bandwidth': 0,
                'cycles': 0
            }
            self.event_log.append(event)

            # Calculate compute metrics
            compute_expense, weights = node.get_stats()
            time_compute = compute_expense / config["mm_compute_per_cycle"]
            
            # Memory operations
            read_access = node.mem_fetch
            write_access = 0
            self.mem_read_access[1] += weights

            assert self.mem_util[0] <= self.mem_size[0]
            self.mem_util[0] += node.in_edge_mem
            node.mem_util = node.out_edge_mem + node.mem_fetch
            
            # Total Free memory
            for i in range(self.mle - 1):
                self.mem_free[i] = self.mem_size[i] - self.mem_util[i]
                
            read_bw_ll = read_access / time_compute
            write_bw_ll = write_access / time_compute
            step_cycles = time_compute
            
            read_bw_req.append(read_bw_ll)
            write_bw_req.append(write_bw_ll)
            free_cycles.append(step_cycles)
            n_swaps = 1
            total_mem = 0

            # Memory management
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
                swap_time = max(config["mm_compute_per_cycle"] * 4, time_compute // n_swaps)
                
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

            self.mem_util_log.append(self.mem_util[0])
            self.mem_read_access[0] += node.weights + node.out_edge_mem
            self.mem_write_access[0] += node.weights + node.out_edge_mem
            
            # Check bandwidth availability
            bandwidth_available = read_bw_ll < self.mem_read_bw[self.mle - 1]

            # If Bandwidth is not available: Cannot Prefetch
            if not bandwidth_available:
                step_cycles += (
                    read_bw_ll / self.mem_read_bw[self.mle - 1] - 1
                ) * time_compute
                self.bandwidth_idle_time += (
                    read_bw_ll / self.mem_read_bw[self.mle - 1] - 1
                ) * time_compute

            # Prefetch using FX graph edges
            if self.mem_free[0] > 0 and bandwidth_available:
                # Get successors from FX graph edges
                successors = graph.get_node_successors(node)
                for next_node in successors:
                    if self.mem_free[0] > next_node.mem_fetch:
                        read_access += next_node.mem_fetch
                        if read_access / step_cycles < self.mem_read_bw[self.mle - 1]:
                            self.mem_util[0] += next_node.mem_fetch
                            self.mem_free[0] -= next_node.mem_fetch
                            next_node.mem_fetch = 0
                        else:
                            read_access = self.mem_read_bw[self.mle - 1] * step_cycles
                            self.mem_util[0] += read_access - read_bw_ll * step_cycles
                            self.mem_free[0] -= read_access - read_bw_ll * step_cycles
                            next_node.mem_fetch -= read_access - read_bw_ll * step_cycles
                    else:
                        read_access += self.mem_free[0]
                        if read_access / step_cycles < self.mem_read_bw[self.mle - 1]:
                            next_node.mem_fetch = next_node.mem_fetch - self.mem_free[0]
                            self.mem_util[0] = self.mem_size[0]
                            self.mem_free[0] = 0
                        else:
                            read_access = self.mem_read_bw[self.mle - 1] * step_cycles
                            self.mem_util[0] += read_access - read_bw_ll * step_cycles
                            self.mem_free[0] -= read_access - read_bw_ll * step_cycles
                            next_node.mem_fetch -= read_access - read_bw_ll * step_cycles

            self.mem_util_full.append(self.mem_util[0])

            if mem_free:
                self.mem_util[0] -= node.out_edge_mem + node.weights + node.in_edge_mem

            # Log node completion
            self.logger.debug(
                "Node operator %r, Compute Expense %d, Time Compute %d, Step Cycles %d, Read Accesses %d, Write Accesses %d, No of Swaps %d, Total_mem %d",
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

            # Log completion state
            event = {
                'step': n,
                'node': node.operator,
                'phase': 'complete',
                'compute_util': compute_expense / (config["mm_compute_per_cycle"] * step_cycles),
                'mem_util': self.mem_util[0] / self.mem_size[0],
                'bandwidth': read_access / step_cycles,
                'cycles': step_cycles
            }
            self.event_log.append(event)

            # Store the last node's out_edge_mem for use after the loop
            last_node_out_edge_mem = node.out_edge_mem

        self.mem_write_access[1] += last_node_out_edge_mem
        return (
            read_bw_req,
            write_bw_req,
            read_bw_actual,
            write_bw_actual,
            cycles,
            free_cycles,
        )
