class Mapper:
    def __init__(self, opts=None, constrainsts=None):
        total_cycles = 0

    def run(self, graph):
        """
        """
        for nodes in graph.nodes():
            # Also check how to schedule next node
            read_access, write_access, memory_bandwidth = node_memory_statistics(node)
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

    def node_memory_statistics(self):
        read_access = 0
        write_access = 0
        memory_bandwidth = 0
        nodememory(node.type, node.input, node.output)
        return read_access, write_access, memory_bandwidth

    def get_number_cycles(self, ismemorybottleneck, *args, **kwargs):
        if ismemorybottleneck != True:
            return node.computational_expense // hwdesc
        else:
            return total_memory_accesses // memory_bandwidth

