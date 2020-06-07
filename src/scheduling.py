class Scheduling:
    def __init__(self, opts=None, constrainsts=None):
        total_cycles = 0

    # Start with a random Hardware Point
    # Do the Scheduling with that Point
    # Check the Bottlenecks and Update the Hardware Accordingly

    def run(self, graph):
        """
        """
        for node in graph.nodes:
            # Also check how to schedule next node
            read_access, write_access, memory_bandwidth = node.get_stats()
            if memory_bandwidth < maxconstraints.bandwidth:
                step_cycles = get_number_cycles(node, True)
            logger.save(
                node,
                step_cycles,
                read_access,
                write_access,
                memory_bandwidth,
                ismemorybottleneck,
            )
            total_cycles += step_cycles
        logger.save(total_cycles)

    def get_number_cycles(self, ismemorybottleneck, *args, **kwargs):
        if ismemorybottleneck != True:
            return (
                node.computational_expense // maxconstraints.primitive.compute_expense
            )
        else:
            return total_memory_accesses // memory_bandwidth
