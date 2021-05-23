lib_hw_dict = ["add", "mult", "buffer", "reg", "sys_array", "logic", "fsm"]

op2sym_map = {
    'And': 'and', 'Or': 'or',
    'Add': '+', 'Sub': '-', 'Mult': '*', 'FloorDiv': '//', 'Mod': '%',
    'LShift': '<<', 'RShift': '>>',
    'BitOr': '|', 'BitXor': '^', 'BitAnd': '&',
    'Eq': '==', 'NotEq': '!=', 'Lt': '<', 'LtE': '<=', 'Gt': '>', 'GtE': '>=',
    'IsNot': '!=',
    'USub': '-', 'UAdd': '+', 'Not': '!', 'Invert': '~'
}
import astor


def parser(graph):
    """
    Parse a non-AI workload graph and store the configuration as a hardware representation 
    """
    # G_hw_repr(graph)
    lib_common_ops = ["add", "mult", "mmv", "spmv", "conv"]
    common_ops = set()
    # if node.operator in lib_common_ops:
    #     common_ops.add(node.operator)
    for node in graph:
        variables = {}
        for i in node.statements:     
                print(i)
                if type(i)==ast.Import:
                    continue
                if type(i)==ast.FunctionDef:
                    string = i.body[0].value
                    print(string)
                    if type(string) == ast.BinOp:
                        print(astor.to_source(string))
                    if(type(string) == ast.BoolOp):
                        print(astor.to_source(string))
                    if (type(string) == ast.Call):
                        print(astor.to_source(string))
                    if (type(string) == ast.Constant):
                        print(astor.to_source(string))
                
                if type(i)==ast.Assign:
                    line = (astor.to_source(i))
                    print(line)
                if type(i)==ast.If:
    #                 var[i.targets.Name.id] = i.Value
                    print(astor.to_source(i.test))
    #                 print(ast.dump(i))
                if type(i)==ast.Return:
    #                 var[i.targets.Name.id] = i.Value
                    print(astor.to_source(i.value))   
                if(type(i) == ast.BinOp):
                        print(astor.to_source(i))
                if(type(i) == ast.BoolOp):
                        print(astor.to_source(i))
                if (type(i) == ast.For):
                        print("printing Loop")
                        print(astor.to_source(i))
            
                                      
                    
hw_allocated = {}

def allocation(H):
    for node in graph:
        hw_allocated[node.name] = allocate(node)
    
    # conflict graph
    
    
    # interval graph for registers
    return hw_allocated

def get_params(dfg, area_budget):
    allocated_area = 0
    while (allocated_area < 0.9*area or allocated_area > 1.2*area):
        # unroll_params -> modify
        # memory size -> modify
        if (area > allocated_area):
            for param in unroll_params:
                # decrease parallelism
                # unroll_params --
                pass
            for mem_cfgs in memory_cfgs:
                # high registers to sram
                # decreases bandwidth
                # update_memory_cfgs 
                pass
        # if(area < allocated_area):
    pass


def allocate_node(node):
    unroll_params = {}
    memory_cfgs = {}
    if(node.operator == 'if_else'):
        return create_fsm(node)
    if(node.operator=='loop'):
        #unroll
        # transform
        # check
        pass
    if(node.operator=='func'):
        getall = []
        for i in func:
            getall.append(allocate_node(i))
        return getall

def create_fsm():
      # fsm overhead and resource consumption
    pass


def create_datapath():
    # cycle time
    # functional units packing < clock cycle
    # datapath, scratchpad access and memory access
    # call datapath optimizations
    # step through memory accesses
    # find common datapath of instances, or uncommon datapath -> area constraint controlled
    pass


def optimizations():
    pass
    # initBaseAddress
    # for each variable allocated assign a base address
    # writeBaseAddress
    # write base address to directory
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


def scheduling(cfg):
    # rescheduleNodesWhenNeeded : (ALAP) rescheduling for non-memory, non-control nodes.
    # upsamplelloops
    # run
    pass


def get_stats(cfg):

    # Write logs
    # * cycle_num,num-of-muls,num-of-adds,num-of-bitwise-ops,num-of-reg-reads,num-of-reg-writes
    #  * If it is called from ScratchpadDatapath, it also outputs per cycle memory
    #  * activity for each partitioned array. add up all the activity of all the components to get the fina
    ddg = {}
    cycles = 0
    print("-------------------------------")
    print("Generating DDDG")          
    print("Num of Nodes: 14811")
    print("Num of Edges: 35693")
    print("Num of Reg Edges: 19191")
    print("Num of MEM Edges: 2132")
    print( "Num of Control Edges: 14810")
    print("Num of Nodes:",  ddg['nodes'])
    print("Num of Edges:",  ddg['edges'])
    print("Num of Reg Edges:", regedges)
    print("Num of MEM Edges:", memedges)
    print("Num of Control Edges:", controledges)
    print("Creating Base Data Path")
    print("Cycle :", cycles)
    print("Avg Power :", avgpower)   

    print("Avg FU Power :", fupower)
    print("Avg FU Dynamic Power:",fu_dynamic_power)
    print("Avg FU leakage Power: ", fu_leakage_power )
    print("Avg MEM Power: ", mempower)
    print("Avg MEM Dynamic Power: ", mem_dynamic_power)
    print("Avg MEM Leakage Power: ", mem_leakage_power)
    print("Area Calculation :", area)

    print("Number of Muxes :", muxes)
    
## choices for scheduling :
## assumptions for our formulas : propagation of error 


# lib_template_space = ["global_mem", "local_mem", "pes", "noc", "buffers"]


# def template_space(H):
#     template_space = {}
#     for i in lib_template_space:
#         template_space[i] = template_handlers(i, hw_allocated)


# def template_handlers(i, hw_allocated):
#     return hw_allocated.gather(i)
