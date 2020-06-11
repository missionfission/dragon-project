import collections

import yaml
import yamlordereddictloader

from utils.logger import create_logger


class Scheduling:
    def __init__(
        self,
        prim_file="primitive.yaml",
        constraintfiles=["max.yaml", "min.yaml"],
        hwfile="defaulthw.yaml",
        opts=None,
    ):
        base_dir = "configs/"
        self.total_cycles = 0

        self.maxval = yaml.load(
            open(base_dir + constraintfiles[0]), Loader=yamlordereddictloader.Loader
        )
        self.minval = yaml.load(
            open(base_dir + constraintfiles[1]), Loader=yamlordereddictloader.Loader
        )
        self.primitives = yaml.load(
            open(base_dir + prim_file), Loader=yamlordereddictloader.Loader
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
        execution_logger = create_logger("logs/execution.txt")
        stats_logger = create_logger("logs/stats.txt")
        for node in graph.nodes:
            compute_expense, read_access, write_access = node.get_stats()
            execution_logger.info("Execution Node %r", node)
            # what will be time taken in compute
            time_compute = compute_expense / config["compute"]
            read_bw = read_access / time_compute
            write_bw = write_access / time_compute
            if read_bw < config["read_bw"] or write_bw < config["write_bw"]:
                execution_logger.info("Node has Memory Bottleneck %r", True)
                step_cycles = time_compute
                # Check the Data Dependence Graph and Prefetch more nodes bandwidth
            elif read_bw < config["read_bw"] and write_bw > config["write_bw"]:
                step_cycles = write_bw / config["write_bw"]
            elif read_bw > config["read_bw"] and write_bw < config["write_bw"]:
                step_cycles = read_bw / config["read_bw"]
            else:
                step_cycles = max(
                    read_bw / config["read_bw"], write_bw / config["write_bw"]
                )
            stats_logger.info(
                "%r %d %d %d %d %d",
                node,
                step_cycles,
                read_access,
                write_access,
                read_bw,
                write_bw,
            )
            self.total_cycles += step_cycles
        stats_logger.info("Total No of cycles  = %d ", self.total_cycles)

    def create_config(self, hwdict):
        config = collections.OrderedDict()
        return config
