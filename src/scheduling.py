class Scheduling:
    def __init__(self, maxval=None, minval=None, opts=None):
        total_cycles = 0
        self.maxval = maxval

    # Start with a random Hardware Point
    # Do the Scheduling with that Point
    # Check the Bottlenecks and Update the Hardware Accordingly

    def set_primitives(self, compute_primitve, memory_primitive):
        self.compute_primitve = compute_primitve
        self.memory_primitive = memory_primitive

    def run(self, graph):
        """
        """
        for node in graph.nodes:
            # Also check how to schedule next node
            read_access, write_access, compute_expense = node.get_stats()
            for i in range(self.maxval["levels"]):
                if bw_req < self.maxval[""]:
                    step_cycles = get_number_cycles(node, True)

            stats_logger.info(
                "%d %d %d %d %d", node, step_cycles, read_access, write_access, bw_req
            )
            stats_logger.info("Node has Memory Bottleneck %b",)

            total_cycles += step_cycles

        stats_logger.info("No of cycles %d-", total_cycles)

    def get_number_cycles(self, ismemorybottleneck, *args, **kwargs):
        if ismemorybottleneck != True:
            return (
                node.computational_expense
                // maxval[self.compute_primitive]["compute_expense"]
            )
        else:
            return total_memory_accesses // memory_bandwidth
