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
        Compute is Excessive at Some Nodes ?
        Do a Hardware Pass and Try to Small Nodes Together ?
        Memory State = Utilization, Bandwidth Use
        Memory Statistics = Size and Maximum Bandwidth 
        Scenario 1 : Utilization is high/low, Bandwidth Use is High ?
        Then : Cannot Prefetch
        Scenario 2 : Utilization is High, Bandwidth Use is Low ?
        Then : Stream Current Node
        Scenario 3 : Utilization is Low, Bandwidth Use is Low ?
        Then : Can Prefetch Next Node, Read access of the next node will decrease
        """

        config = self.config

        graph = self.merge_nodes(graph)

        for node in graph.nodes:
            compute_expense, read_access, write_access = node.get_stats()
            self.logger.info(node.get_stats())
            # what will be time taken in compute
            time_compute = compute_expense / config["mm_compute_per_cycle"]
            read_bw_ll = read_access / time_compute
            write_bw_ll = write_access / time_compute
            if (
                read_bw_ll < self.mem_read_bw[self.mle - 1]
                and write_bw_ll < self.mem_write_bw[self.mle - 1]
            ):
                # Last level memory fetch takes more time, so that may be a bottleneck
                self.logger.info("node has memory bottleneck at level  %d", 0)
                self.logger.info("memory size is too small  %d", 0)
                # Check the Data Dependence Graph and Prefetch more nodes bandwidth
                if self.mem_util[0] < self.mem_size[0]:
                    self.mem_util[0] += self.prefetch(node.next)
                step_cycles = time_compute

            elif (
                read_bw_ll < self.mem_read_bw[self.mle - 1]
                and write_bw_ll > self.mem_write_bw[self.mle - 1]
            ):
                step_cycles = write_bw_ll / self.mem_write_bw[self.mle - 1]

            elif (
                read_bw_ll > self.mem_read_bw[self.mle - 1]
                and write_bw_ll < self.mem_write_bw[self.mle - 1]
            ):
                step_cycles = read_bw_ll / self.mem_read_bw[self.mle - 1]

            else:
                step_cycles = max(
                    read_bw_ll / self.mem_read_bw[self.mle - 1],
                    write_bw_ll / self.mem_write_bw[self.mle - 1],
                )
            self.logger.info(
                "Node operator %r, Step Cycles %d, Read Accesses %d, Write Accesses %d ",
                node.operator,
                step_cycles,
                read_access,
                write_access,
            )
            self.total_cycles += step_cycles

    def merge_nodes(self, graph):
        # Check nodes which are independent and parallel, can be merged ?
        ## If the nodes combined compute is lesser than total available compute
        nodes = graph.nodes
        if node1.compute_expense + node2.compute_expense < total_compute:
            node3 = [node1, node2]

        self.logger.info("Merging Nodes %r %r", node1.operator, node2.operator)
        self.logger.info(
            "Merging Nodes %d %d", node1.compute_expense, node2.compute_expense
        )
        self.logger.info("")

        return graph

    def create_config(self, hwdict):
        config = hwdict["architecture"]

        self.logger.info("Config Statistics : ")

        self.mle = config["memory_levels"]

        self.read_accesses = np.zeros((self.mle))
        self.write_accesses = np.zeros((self.mle))
        self.mem_size = np.zeros((self.mle))
        self.mem_util = np.zeros((self.mle))
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

    def prefetch(self, node):
        return 0
