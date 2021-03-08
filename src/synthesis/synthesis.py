lib_hw_dict = ["add", "mult", "buffer", "reg", "sys_array", "logic", "fsm"]


def parser(graph):
    """
    Parse a non-AI workload graph and store the configuration as a hardware representation 
    """
    # G_hw_repr(graph)
    lib_common_ops = ["add", "mult", "mmv", "spmv", "conv"]
    common_ops = set()
    for node in graph:
        if node.operator in lib_common_ops:
            common_ops.add(node.operator)


hw_allocated = {}

# def allocation(H):
#     for node in graph:
#         hw_allocated[node.name] = allocate(node)

# def binding(H):
#     for common_ops in graph:
#         for key in hw_allocated.keys():
#             if(key in common_ops):
#                 hw_allocated["common_ops"] = merge(hw_allocated[key], hw_allocated["common_ops"].copy())

# def allocate_node(node):
#     if(node.operator == 'if_else'):
#         return create_fsm(node)
#     if(node.operator=='loop'):
#         #unroll
#         # transform
#         # check
#         pass
#     if(node.operator=='func'):
#         getall = []
#         for i in func:
#             getall.append(allocate_node(i))
#         return getall

# def create_fsm():
#       # fsm overhead and resource consumption
#     pass


def create_datapath():
    # cycle time
    # functional units packing < clock cycle
    # datapath, scratchpad access and memory access
    # call datapath optimizations
    # step through memory accesses
    # find common datapath of instances, or uncommon datapath -> area constraint controlled
    pass


lib_template_space = ["global_mem", "local_mem", "pes", "noc", "buffers"]


def template_space(H):
    template_space = {}
    for i in lib_template_space:
        template_space[i] = template_handlers(i, hw_allocated)


def template_handlers(i, hw_allocated):
    return hw_allocated.gather(i)


def optimizations():
    pass
    # initBaseAddress
    # writeBaseAddress
    # initDmaBaseAddress
    # memoryAmbiguation
    # removePhiNodes
    # loopFlatten, loopUnrolling : Loop Tree
    # removeInductionDependence
    # GloballoopPipelining, perLoopPipelining
    # fuseRegLoadStores, fuseConsecutiveBranches, removeSharedLoads : LoadBuffering
    #  updateGraphWithIsolatedEdges(to_remove_edges);
    #  updateGraphWithNewEdges(to_add_edges);
    # storeBuffer, removeRepeatedStores, treeHeightReduction


class graph_manipulations:
    def __init__(self, graph):
        self.graph = graph

    def to_remove_edges(self):
        pass

    def to_add_edges(self):
        pass

    def isolated_nodes(self):
        pass

    def isolated_edges(self):
        pass

    def dependency_nodes(self):
        pass


def scheduling():
    # rescheduleNodesWhenNeeded : (ALAP) rescheduling for non-memory, non-control nodes.
    # upsamplelloops
    # run
    pass


def get_stats():

    # Write logs
    # * cycle_num,num-of-muls,num-of-adds,num-of-bitwise-ops,num-of-reg-reads,num-of-reg-writes
    #  * If it is called from ScratchpadDatapath, it also outputs per cycle memory
    #  * activity for each partitioned array. add up all the activity of all the components to get the final
    # output
    pass
