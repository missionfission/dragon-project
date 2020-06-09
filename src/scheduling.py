class Scheduling:
    def __init__(
        self,
        primitive="primtive.yaml",
        constraintfiles=["max.yaml", "min.yaml"],
        hwfile="default.yaml",
        opts=None,
    ):
        total_cycles = 0
        base_dir = "configs/"
        self.maxval = yaml.load(
            open(base_dir + constraintfiles[0]), Loader=yamlordereddictloader.Loader
        )
        self.minval = yaml.load(
            open(base_dir + constraintfiles[1]), Loader=yamlordereddictloader.Loader
        )
        self.primitives = yaml.load(
            open(base_dir + primitive_file), Loader=yamlordereddictloader.Loader
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
        for node in graph.nodes:
            read_access, write_access, compute_expense = node.get_stats()
            execution_logger.info("Execution Node %d", node)
            if bw_req < self.maxval[""]:
                execution_logger.info("Node has Memory Bottleneck %b", True)
                step_cycles = get_number_cycles(node, True)
            else:
                step_cycles = get_number_cycles(node, False)

            stats_logger.info(
                "%d %d %d %d %d", node, step_cycles, read_access, write_access, bw_req
            )
            total_cycles += step_cycles

        stats_logger.info("No of cycles %d - ", total_cycles)

    def get_number_cycles(self, membtlnck, *args, **kwargs):
        if membtlnck != True:
            return (
                node.computational_expense // maxval[self.compute_primitive]["latency"]
            )
        else:
            return total_memory_accesses // memory_bandwidth
