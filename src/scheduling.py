import collections

import yaml
import yamlordereddictloader

from utils.logger import create_logger


class Scheduling:
    def __init__(
        self,
        constraintfiles=["max.yaml", "min.yaml"],
        hwfile="default.yaml",
        opts=None,
    ):
        base_dir = "configs/"
        self.total_cycles = 0
        self.logger = create_logger("logs/stats.txt")

        self.maxval = yaml.load(
            open(base_dir + constraintfiles[0]), Loader=yamlordereddictloader.Loader
        )
        self.minval = yaml.load(
            open(base_dir + constraintfiles[1]), Loader=yamlordereddictloader.Loader
        )
        self.config = self.create_config(
            yaml.load(open(base_dir + hwfile), Loader=yamlordereddictloader.Loader)
        )

    def run(self, graph):
        """
        Scheduling works in the following way :
        1. Start with a given/random Hardware point -> Nodes of the graph are scheduled (prefetching)
        2. Do the Scheduling with that Point -> Mapping stops here -> Further evaluation is done using accelergy 
        (with values taken from ERT/ART) -> If values not available -> Use plugins for generating these values 
        3. Log bottlenecks and work on a different Hardware point -> do this till some realistically
        max, min values are not violated -> Values/Analyses for a different/unavailable point will require full 
        integration of plugins -> Currently using a table at 40nm.
        """
        config = self.config
        mem_util = np.zeros((self.mle))

        # Read access of the next node will decrease
        # Check both size, utilization and bandwidths at every node
        # What about memory size that can also get exhausted ?
        # So if memory size is exhausted, then have to go to a previous level and write there ?
        # if any level utilization is exhausted then only the immediate memory required will be kept.
        # if the memory is empty in size, but is not bandwidth, it is useless?
        # Cannot do prefetching, if it is empty in both
        # size and bandwidth, can do prefetching, but now the memory fetches have to check,
        # whether to do fetches of the same
        # node or a different node
        # say bandwidth at level0 is sufficient, at level1 is insufficient, then at level1 we have a bottlenecks
        # slower so it will take its own time
        # Do vector operations in the meantime perhaps ?

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
                step_cycles = write_bw / self.mem_write_bw[self.mle - 1]

            elif (
                read_bw_ll > self.mem_read_bw[self.mle - 1]
                and write_bw_ll < self.mem_write_bw[self.mle - 1]
            ):
                step_cycles = read_bw / self.mem_read_bw[self.mle - 1]

            else:
                step_cycles = max(
                    read_bw / self.mem_read_bw[self.mle - 1],
                    write_bw / self.mem_write_bw[self.mle - 1],
                )
            self.logger.info(
                "Node operator %r, Step Cycles %d, Read Accesses %d, Write Accesses %d ",
                node.operator,
                step_cycles,
                read_access,
                write_access,
            )
            self.total_cycles += step_cycles
        self.logger.info("Total No of cycles  = %d ", self.total_cycles)

    def create_config(self, hwdict):
        config = hwdict["architecture"]
        self.logger.info("Config Statistics : ")

        self.mle = config["memory_levels"]
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
