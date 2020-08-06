import collections

import numpy as np
import yaml
import yamlordereddictloader

from utils.logger import create_logger


class Scheduling:
    def __init__(self, hwfile="default.yaml"):
        base_dir = "configs/"
        self.total_cycles = 0
        self.logger = create_logger("logs/stats.txt")
        self.config = self.create_config(
            yaml.load(open(base_dir + hwfile), Loader=yamlordereddictloader.Loader)
        )

    def run(self, graph):

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

        read_bw_req = []
        write_bw_req = []
        read_bw_actual = []
        write_bw_actual = []
        cycles = []
        transferable_checkpointed_edge = []
        all_checkpointed_edge = []
        # Mem Fetch time of the last Nodes

        for n, node in enumerate(graph.nodes):

            # These are last level read/write accesses
            compute_expense, read_access, write_access = node.get_stats()

            self.logger.info(node.get_stats())
            self.mem_util[0] += node.in_edge_mem

            # Total Free memory
            for i in range(self.mle - 1):
                self.mem_free[i] = self.mem_size[i] - self.mem_util[i]

            time_compute = compute_expense / config["mm_compute_per_cycle"]
            read_bw_ll = read_access / (2 * time_compute)
            write_bw_ll = write_access / (2 * time_compute)
            step_cycles = 2 * time_compute

            if self.mem_free[0] < node.mem_util:
                mem_free = True
                # node mem_util = output edge
                self.logger.info("Memory size is too low/ Memory is Full")
                self.logger.info("Node or Node memory Requirements too high")
                # Rearrange the checkpointed_nodes
                #                     rearrange = True

                # Is it possible now : Otherwise update the last level memory bandwidth requirements
                step_cycles += (node.mem_util // self.mem_free[0] + 1) * (
                    self.mem_free[0] / self.mem_read_bw[self.mle - 1]
                )
                read_bw_ll = read_access / step_cycles
                write_bw_ll = write_access / step_cycles

            else:
                self.mem_util[0] += node.mem_util
                self.mem_free[0] -= node.mem_util

            read_bw_req.append(read_bw_ll)
            write_bw_req.append(write_bw_ll)

            # Last level memory fetch takes more time, so that may be a bottleneck
            bandwidth_available = read_bw_ll < self.mem_read_bw[self.mle - 1]

            # If Bandwidth is not available : Cannot Prefetch
            if (bandwidth_available) == False:
                step_cycles += (
                    read_bw_ll / self.mem_read_bw[self.mle - 1]
                ) * time_compute

            # If memory is not free for the next node and Bandwidth is available : Move nodes back and forth
            # if(total_mem_free[0] == 0 and (bandwidth_available)):
            # for(nodes in checkpointed_nodes):
            # checkpointed but not immediate node

            # Check if memory is free and Bandwidth available : From the Data Dependence Graph, Prefetch new node
            if self.mem_free[0] > 0 and (bandwidth_available):
                # print(n,node.next)
                if n < len(graph.nodes) - 1:
                    if self.mem_free[0] > node.next.mem_util:
                        read_access += node.next.mem_util
                        if read_access / step_cycles < self.mem_read_bw[self.mle - 1]:
                            self.mem_util[0] += node.next.mem_util
                            self.mem_free[0] -= node.next.mem_util
                            node.next.mem_util = 0
                        else:
                            read_access = self.mem_read_bw[self.mle - 1] * step_cycles
                            self.mem_util[0] += read_access - read_bw_ll * step_cycles
                            self.mem_free[0] -= read_access - read_bw_ll * step_cycles
                            node.next.mem_util = read_access - read_bw_ll * step_cycles

                    else:
                        read_access += self.mem_free[0]
                        if read_access / step_cycles < self.mem_read_bw[self.mle - 1]:
                            node.next.mem_util = node.next.mem_util - self.mem_free[0]
                            self.mem_util[0] = self.mem_size[0]
                            self.mem_free[0] = 0
                        else:
                            read_access = self.mem_read_bw[self.mle - 1] * step_cycles
                            self.mem_util[0] += read_access - read_bw_ll * step_cycles
                            self.mem_free[0] -= read_access - read_bw_ll * step_cycles
                            node.next.mem_util = read_access - read_bw_ll * step_cycles

                # TODO Consider Write bandwidth for a block read memory or Write Bandwidth  for a endurance purposes
            self.mem_util[0] -= node.in_edge_mem

            if mem_free:
                self.mem_util[0] -= node.mem_util

            self.logger.info(
                "Node operator %r, Step Cycles %d, Read Accesses %d, Write Accesses %d ",
                node.operator,
                step_cycles,
                read_access,
                write_access,
            )
            self.total_cycles += step_cycles
            print(self.mem_free[0], self.mem_util[0], self.mem_size[0])
            cycles.append(step_cycles)
            read_bw_actual.append(read_access / step_cycles)
            write_bw_actual.append(write_access / step_cycles)
        return read_bw_req, write_bw_req, read_bw_actual, write_bw_actual, cycles

    def create_config(self, hwdict):
        config = hwdict["architecture"]

        self.logger.info("Config Statistics : ")

        self.mle = config["memory_levels"]

        self.read_accesses = np.zeros((self.mle))
        self.write_accesses = np.zeros((self.mle))
        self.mem_size = np.zeros((self.mle))
        self.mem_util = np.zeros((self.mle))
        self.mem_free = np.zeros((self.mle))
        self.mem_read_bw = np.zeros((self.mle))
        self.mem_write_bw = np.zeros((self.mle))

        mm_compute = config["mm_compute"]
        vector_compute = config["vector_compute"]

        if config["mm_compute"]["class"] == "systolic_array":
            config["mm_compute_per_cycle"] = (
                ((mm_compute["size"]) ** 2) * mm_compute["N_PE"] / 2
            )
            config["comp_bw"] = (
                mm_compute["size"] * mm_compute["N_PE"] * mm_compute["frequency"] * 2
            )

            self.logger.info(
                "MM Compute per cycle : %d", config["mm_compute_per_cycle"]
            )
            self.logger.info("Compute Bandwidth Required : %d", config["comp_bw"])

        if config["mm_compute"]["class"] == "mac":
            config["mm_compute_per_cycle"] = (
                ((mm_compute["size"]) ** 2) * mm_compute["N_PE"] / 2
            )
            config["comp_read_bw"] = (
                mm_compute["size"] * mm_compute["N_PE"] * mm_compute["frequency"] * 2
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

            self.logger.info(
                "Memory at Level %d, Read Bandwidth %d Write Bandwidth %d",
                i,
                self.mem_read_bw[i],
                self.mem_write_bw[i],
            )

        return config
