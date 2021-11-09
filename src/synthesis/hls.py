import ast
import re

import astor

from generator import get_mem_props
from ir.cfg.staticfg import CFGBuilder

lib_hw_dict = ["add", "mult", "buffer", "reg", "sys_array", "logic", "fsm"]
common_ops = set()
op2sym_map = {
    "And": "and",
    "Or": "or",
    "Add": "+",
    "Sub": "-",
    "Mult": "*",
    "FloorDiv": "//",
    "Mod": "%",
    "LShift": "<<",
    "RShift": ">>",
    "BitOr": "|",
    "BitXor": "^",
    "BitAnd": "&",
    "Eq": "==",
    "NotEq": "!=",
    "Lt": "<",
    "LtE": "<=",
    "Gt": ">",
    "GtE": ">=",
    "IsNot": "!=",
    "USub": "-",
    "UAdd": "+",
    "Not": "!",
    "Invert": "~",
}
delimiters = (
    "+",
    "-",
    "*",
    "//",
    "%",
    "=",
    ">>",
    "<<",
    "<",
    "<=",
    ">",
    ">=",
    "!=",
    "~",
    "!",
    "^",
    "&",
)
regexPattern = "|".join(map(re.escape, delimiters))

latency = {
    "And": 1,
    "Or": 1,
    "Add": 4,
    "Sub": 4,
    "Mult": 5,
    "FloorDiv": 16,
    "Mod": 3,
    "LShift": 0.70,
    "RShift": 0.70,
    "BitOr": 0.06,
    "BitXor": 0.06,
    "BitAnd": 0.06,
    "Eq": 1,
    "NotEq": 1,
    "Lt": 1,
    "LtE": 1,
    "Gt": 1,
    "GtE": 1,
    "USub": 0.42,
    "UAdd": 0.42,
    "IsNot": 1,
    "Not": 0.06,
    "Invert": 0.06,
    "Regs": 1,
}
energy = {}
power = {
    "And": 32 * [1.010606e-02, 7.950398e-03, 1.805590e-02, 1.805590e-02, 6.111633e-04],
    "Or": 32 * [1.010606e-02, 7.950398e-03, 1.805590e-02, 1.805590e-02, 6.111633e-04],
    "Add": [2.537098e00, 3.022642e00, 5.559602e00, 1.667880e01, 5.311069e-02],
    "Sub": [2.537098e00, 3.022642e00, 5.559602e00, 1.667880e01, 5.311069e-02],
    "Mult": [5.050183e00, 6.723213e00, 1.177340e01, 3.532019e01, 1.198412e-01],
    "FloorDiv": [5.050183e00, 6.723213e00, 1.177340e01, 3.532019e01, 1.198412e-01],
    "Mod": [5.050183e00, 6.723213e00, 1.177340e01, 3.532019e01, 1.198412e-01],
    "LShift": [8.162355e-02, 3.356332e-01, 4.172512e-01, 4.172512e-01, 1.697876e-03],
    "RShift": [8.162355e-02, 3.356332e-01, 4.172512e-01, 4.172512e-01, 1.697876e-03],
    "BitOr": [1.010606e-02, 7.950398e-03, 1.805590e-02, 1.805590e-02, 6.111633e-04],
    "BitXor": [1.010606e-02, 7.950398e-03, 1.805590e-02, 1.805590e-02, 6.111633e-04],
    "BitAnd": [1.010606e-02, 7.950398e-03, 1.805590e-02, 1.805590e-02, 6.111633e-04],
    "Eq": [8.162355e-02, 3.356332e-01, 4.172512e-01, 4.172512e-01, 1.697876e-03],
    "NotEq": [8.162355e-02, 3.356332e-01, 4.172512e-01, 4.172512e-01, 1.697876e-03],
    "Lt": [8.162355e-02, 3.356332e-01, 4.172512e-01, 4.172512e-01, 1.697876e-03],
    "LtE": [8.162355e-02, 3.356332e-01, 4.172512e-01, 4.172512e-01, 1.697876e-03],
    "Gt": [8.162355e-02, 3.356332e-01, 4.172512e-01, 4.172512e-01, 1.697876e-03],
    "GtE": [8.162355e-02, 3.356332e-01, 4.172512e-01, 4.172512e-01, 1.697876e-03],
    "USub": [1.010606e-02, 7.950398e-03, 1.805590e-02, 1.805590e-02, 6.111633e-04],
    "UAdd": [1.010606e-02, 7.950398e-03, 1.805590e-02, 1.805590e-02, 6.111633e-04],
    "IsNot": [8.162355e-02, 3.356332e-01, 4.172512e-01, 4.172512e-01, 1.697876e-03],
    "Not": [1.010606e-02, 7.950398e-03, 1.805590e-02, 1.805590e-02, 6.111633e-04],
    "Invert": [1.010606e-02, 7.950398e-03, 1.805590e-02, 1.805590e-02, 6.111633e-04],
    "Regs": [7.936518e-03, 1.062977e-03, 8.999495e-03, 8.999495e-03, 7.395312e-05],
}

hw_allocated = {}
memory_cfgs = {}
hw_utilized = {}

cycles = 0
hw_allocated["Regs"] = 0
hw_utilized["Regs"] = 0


def schedule(expr, type):
    """[Schedules the expr from AST]

    Args:
        expr (): 
        type (): 

    Returns:
        : 
    """
    # rescheduleNodesWhenNeeded : (ALAP) rescheduling for non-memory, non-control nodes.
    # upsamplelloops
    # run
    hw_need = {}
    data_state = False
    mem_state = False
    bw_avail = 0
    bw_req = np.MAX
    num_cycles = 0
    for key in op2sym_map.keys():
        hw_need[key] = 0
    strs = re.split(regexPattern, expr)
    print(strs, expr)
    if strs.count("") > 0:
        strs.remove("")
    num_vars = len(strs)
     # ALAP
    for i, op in enumerate(op2sym_map.values()):
        hw_need[list(op2sym_map.keys())[i]] += expr.count(op)
        num_cycles += hw_need[list(op2sym_map.keys())[i]]*latency[list(op2sym_map.keys())[i]] 
    # ASAP
    
    # Memory Bandwidth Req
    while (data_state and bw_avail):
        num_cycles +=  hw_need[list(op2sym_map.keys())[i]]*latency[list(op2sym_map.keys())[i]]
    # Bandwidth-Rearrangements : Get op Control-Data-Flow
        if bw_req < bw_avail and mem_state == False:
            pass
    #     num_cycles =
    hw_need["Regs"] = num_vars
    return num_cycles, hw_need


def parse_code(expr, type, unrolled=1, loop_iters=1):
    """[Parse the input Python Code file]

    Args:
        expr (): 
        type (): 
        unrolled (int, optional): . Defaults to 1.
        loop_iters (int, optional): . Defaults to 1.
    """
    if type in ["assign", "expr", "binop_nested", "constant"]:
        expr_cycles, hw_need = schedule(expr, type)
        global cycles, hw_allocated, hw_utilized
        cycles += expr_cycles * (int(loop_iters) / int(unrolled))
        # hw_allocated = max(hw_need*unrolled, hw_allocated)
        hw_allocated = {
            key: max(value, hw_need[key] * unrolled)
            for key, value in hw_allocated.items()
        }
        hw_utilized = {
            key: value + hw_need[key] * unrolled for key, value in hw_utilized.items()
        }
        print(cycles)
    # if(type == "assign"):
    #     left, right = expr.split("=")
    #     parse_code(left, "expr")
    #     parse_code(right, "expr")
    # if(type == "expr"):
    #     for i,op in enumerate(op2sym_map.values()):
    #         hw_allocated[list(op2sym_map.keys())[i]] += expr.count(op)
    #     # expression is list operation such as append
    #     # expr.split(" ")
    #     # find brackets, create data path
    # if(type == "binop_nested"):
    #     for i,op in enumerate(op2sym_map.values()):
    #         hw_allocated[list(op2sym_map.keys())[i]] += expr.count(op)
    # if (type == "binop_simple"):
    #     for i,op in enumerate(op2sym_map.values()):
    #         hw_allocated[list(op2sym_map.keys())[i]] += expr.count(op)
    # if (type == "constant"):
    #     for i,op in enumerate(op2sym_map.values()):
    #         hw_allocated[list(op2sym_map.keys())[i]] += expr.count(op)


def check_and_parse(string, unrolled=1, loop_iters=1):
    """

    Args:
        string (): 
        unrolled (int, optional): . Defaults to 1.
        loop_iters (int, optional): . Defaults to 1.
    """
    if type(string) == ast.BinOp or ast.BoolOp:
        parse_code(astor.to_source(string), "binop_nested", unrolled, loop_iters)
    if type(string) == ast.Call:
        # cycles += visit(string)
        # latency calculation of traversal
        pass
    if type(string) == ast.Constant:
        parse_code(astor.to_source(string), "constant", unrolled, loop_iters)


def parse_graph(graph):
    """
    Parse a non-AI workload graph and store the configuration as a hardware representation 
    """
    for key in op2sym_map.keys():
        hw_allocated[key] = 0
        hw_utilized[key] = 0
    unroll_params = {}
    variables = {}
    global memory_cfgs

    # if node.operator in lib_common_ops:
    #     common_ops.add(node.operator)
    for node in graph:
        # yield(node)
        for i in node.statements:
            print(i)
            if type(i) == ast.Import:
                continue
            if type(i) == ast.FunctionDef:
                for string in i.body:
                    if isinstance(string, ast.For):
                        continue
                    # print(string)
                    check_and_parse(string)
            if type(i) == ast.Assign:
                # allocated memory/registers
                flag = True
                if isinstance(i.value, ast.Tuple):
                    for x in list(i.value.elts):
                        if not isinstance(x, ast.Constant):
                            flag = False
                    if flag:
                        if isinstance(i.targets[0], ast.Name):
                            variables[i.targets[0].id] = len(list(i.value.elts))
                else:
                    parse_code(astor.to_source(i), "assign")
            if type(i) == ast.AugAssign:
                # allocated memory/registers
                parse_code(astor.to_source(i), "assign")
            if type(i) == ast.Expr:
                parse_code(astor.to_source(i), "expr")
            if type(i) == ast.If:
                check_and_parse(i.test)
            if type(i) == ast.Return:
                check_and_parse(i.value)
            if isinstance(i, ast.For):
                print(ast.dump(i))
                if isinstance(i.iter.args[0], ast.Constant):
                    loop_iters = [i.iter.args[0].value]
                    # capture unrolling factor for DSE/ will change Number of Memory Banks
                    unroll_params[str(i)] = loop_iters
                    unrolled = loop_iters
               
#                 if isinstance(i.iter.args[0], ast.Variable):
#                     loop_iters = [i.iter.args[0].value]
#                     print("Loop iters are Variable Initialized/Will be Captured by User input")
                else:
                    print("Loop iters could not be captured")
                    print("Enter Loop iters : ")
                    loop_iters = int(input())
                    print("Enter Unroll Parameters : ")
                    unrolled = int(input())
#                   
#                        loop_iters = 1
#                     unrolled = 1
                print("Loop Iters are", loop_iters)
                print("Unrolled are", unrolled)
                for string in i.body:
                    check_and_parse(string, unrolled, loop_iters)
                # print(ast.dump(i))
                # transform
            # numpy library spmv, dot, conv
    memory_cfgs = variables
    # mem_list =  allocate_memory_cfgs()


def get_params(dfg, area_budget):
    """

    Args:
        dfg (): 
        area_budget (): 
    """
    allocated_area = 0
    while allocated_area < 0.9 * area or allocated_area > 1.2 * area:
        # unroll_params -> modify
        # memory size -> modify
        if area > allocated_area:
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


def allocate_memory_cfgs():
    """[allocate_memory_cfgs]

    Returns:
        : 
    """
    mem_list = {}
    for key, value in memory_cfgs.items():
        if value > 32768:
            mem_list[key] = get_mem_props(value, 32, 1)
        else:
            mem_list[key] = power["Regs"] * value
    return mem_list


def prune_allocator():

    # conflict graph
    # interval graph for registers
    if node.operator == "func":
        getall = []
        for i in func:
            getall.append(allocate_node(i))
    return getall


# def get_fsm_overhead():
#       # fsm overhead and resource consumption
#     pass


# def create_datapath():
#     # cycle time
#     # functional units packing < clock cycle
#     # datapath, scratchpad access and memory access
#     # call datapath optimizations
#     # step through memory accesses
#     # find common datapath of instances, or uncommon datapath -> area constraint controlled
#     pass


# def optimizations():
#     pass
#     # initBaseAddress
#     # for each variable allocated assign a base address
#     # writeBaseAddress
#     # write base address to directory
#     # initDmaBaseAddress
#     # memoryAmbiguation
#     # removePhiNodes
#     # loopFlatten, loopUnrolling : Loop Tree
#     # removeInductionDependence
#     # GloballoopPipelining, perLoopPipelining
#     # fuseRegLoadStores, fuseConsecutiveBranches, removeSharedLoads : LoadBuffering
#     #  updateGraphWithIsolatedEdges(to_remove_edges);
#     #  updateGraphWithNewEdges(to_add_edges);
#     # storeBuffer, removeRepeatedStores, treeHeightReduction


# class graph_manipulations:
#     def __init__(self, graph):
#         self.graph = graph

#     def to_remove_edges(self):
#         pass

#     def to_add_edges(self):
#         pass

#     def isolated_nodes(self):
#         pass

#     def isolated_edges(self):
#         pass

#     def dependency_nodes(self):
#         pass


def get_stats(cfg):

    # Write logs
    # * cycle_num,num-of-muls,num-of-adds,num-of-bitwise-ops,num-of-reg-reads,num-of-reg-writes
    #  * If it is called from ScratchpadDatapath, it also outputs per cycle memory
    #  * activity for each partitioned array. add up all the activity of all the components to get the fina
    ddg = {}
    print("-------------------------------")
    print("Generating DDDG")
    avgpower = 0
    # print("Num of Nodes:",  ddg['nodes'])
    # print("Num of Edges:",  ddg['edges'])
    # print("Num of Reg Edges:", regedges)
    # print("Num of MEM Edges:", memedges)
    # print("Num of Control Edges:", controledges)
    print("Creating Base Data Path")
    print("Cycle :", cycles)
    print("Hardware ")
    for keys in hw_utilized.keys():
        avgpower += power[keys][0] * hw_utilized[keys] * latency[keys] / cycles
    print("Avg Power :", avgpower)
    # print("Avg FU Power :", fupower)
    # print("Avg FU Dynamic Power:",fu_dynamic_power)
    # print("Avg FU leakage Power: ", fu_leakage_power )
    # print("Avg MEM Power: ", mempower)
    # print("Avg MEM Dynamic Power: ", mem_dynamic_power)
    # print("Avg MEM Leakage Power: ", mem_leakage_power)
    # print("Avg REG Power: ", regpower)
    # print("Area Calculation :", area)
    print(hw_allocated, memory_cfgs)


## choices for scheduling :
## assumptions for our formulas : propagation of error


# lib_template_space = ["global_mem", "local_mem", "pes", "noc", "buffers"]


# def template_space(H):
#     template_space = {}
#     for i in lib_template_space:
#         template_space[i] = template_handlers(i, hw_allocated)


# def template_handlers(i, hw_allocated):
#     return hw_allocated.gather(i)


# def allocation(H):
#     for node in graph:
#         hw_allocated[node.name] = allocate(node)


#     return hw_allocated
