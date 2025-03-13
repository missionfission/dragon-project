import ast
import re

import astor
import numpy as np

from generator import get_mem_props
from ir.cfg.staticfg import CFGBuilder

lib_hw_dict = ["add", "mult", "buffer", "reg", "sys_array", "logic", "fsm"]
common_ops = set()
op2sym_map = {
    "Add": "Add",
    "Sub": "Sub",
    "Mult": "Mult",
    "FloorDiv": "FloorDiv",
    "Mod": "Mod",
    "LShift": "LShift",
    "RShift": "RShift",
    "BitOr": "BitOr",
    "BitXor": "BitXor",
    "BitAnd": "BitAnd",
    "Eq": "Eq",
    "NotEq": "NotEq",
    "Lt": "Lt",
    "LtE": "LtE",
    "Gt": "Gt",
    "GtE": "GtE",
    "IsNot": "IsNot",
    "USub": "USub",
    "UAdd": "UAdd",
    "Not": "Not",
    "Invert": "Invert"
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
    "Add": 4,
    "Sub": 4,
    "Mult": 5,
    "FloorDiv": 20,
    "Mod": 20,
    "LShift": 3,
    "RShift": 3,
    "BitOr": 3,
    "BitXor": 3,
    "BitAnd": 3,
    "Eq": 2,
    "NotEq": 2,
    "Lt": 2,
    "LtE": 2,
    "Gt": 2,
    "GtE": 2,
    "IsNot": 2,
    "USub": 2,
    "UAdd": 2,
    "Not": 2,
    "Invert": 2,
    # Memory operations
    "load": 4,
    "store": 4,
    "call": 6,
    "compare": 2
}
energy = {}
power = {
    "Add": [0.1, 0.01],
    "Sub": [0.1, 0.01],
    "Mult": [0.3, 0.03],
    "FloorDiv": [0.5, 0.05],
    "Mod": [0.5, 0.05],
    "LShift": [0.05, 0.005],
    "RShift": [0.05, 0.005],
    "BitOr": [0.05, 0.005],
    "BitXor": [0.05, 0.005],
    "BitAnd": [0.05, 0.005],
    "Eq": [0.02, 0.002],
    "NotEq": [0.02, 0.002],
    "Lt": [0.02, 0.002],
    "LtE": [0.02, 0.002],
    "Gt": [0.02, 0.002],
    "GtE": [0.02, 0.002],
    "IsNot": [0.02, 0.002],
    "USub": [0.02, 0.002],
    "UAdd": [0.02, 0.002],
    "Not": [0.02, 0.002],
    "Invert": [0.02, 0.002],
    # Memory operations
    "load": [0.2, 0.02],
    "store": [0.2, 0.02],
    "call": [0.3, 0.03],
    "compare": [0.02, 0.002]
}

hw_allocated = {}
memory_cfgs = {}
hw_utilized = {}
bw_avail = 0
mem_state = {}
for variable in memory_cfgs.keys():
    mem_state[variable]=False
    print(variable)
cycles = 0
hw_allocated["Regs"] = 0
hw_utilized["Regs"] = 0

def schedule(expr, type, variable=None):
    """Schedules the expr from AST

    Args:
        expr: Expression to schedule
        type: Type of expression
        variable (str, optional): Variable being scheduled. Defaults to None.

    Returns:
        tuple: (num_cycles, mem_cycles, hw_need)
    """
    hw_need = {}
    bw_req = np.inf
    num_cycles = 0
    mem_cycles = 0
    
    # Initialize hardware needs
    for key in op2sym_map.keys():
        hw_need[key] = 0
        
    # Parse expression
    strs = re.split(regexPattern, expr)
    if strs.count("") > 0:
        strs.remove("")
    num_vars = len(strs)
    
    # Count operations
    for i, op in enumerate(op2sym_map.values()):
        count = expr.count(op)
        op_type = list(op2sym_map.keys())[i]
        hw_need[op_type] = count
        
        # Calculate cycles based on operation type and count
        if count > 0:
            # Account for parallel execution when possible
            parallelism = min(count, 8)  # Maximum 8-way parallelism
            num_cycles += (count * latency[op_type]) / parallelism
    
    # Handle memory operations
    if variable and variable in memory_cfgs:
        mem_size = memory_cfgs[variable]
        bw_req = mem_size / num_cycles if num_cycles > 0 else np.inf
        
        if bw_req < bw_avail and not mem_state.get(variable, False):
            # Calculate memory cycles considering burst transfers
            burst_size = 64  # 64-byte cache line
            num_bursts = (mem_size + burst_size - 1) // burst_size
            mem_cycles = num_bursts * (mem_size / bw_avail)
            mem_state[variable] = True
            
    # Allocate registers
    if type == "assign":
        # Need registers for both sides of assignment
        hw_need["Regs"] = num_vars * 2
    else:
        # Need registers for operands and result
        hw_need["Regs"] = num_vars
        
    # Account for loop overhead if present
    if "for" in expr:
        num_cycles += 2  # Loop initialization and increment
        hw_need["Add"] += 1  # Loop counter increment
        hw_need["Lt"] += 1   # Loop condition check
        
    # Account for conditional overhead
    if "if" in expr:
        num_cycles += 1  # Condition evaluation
        hw_need["Lt"] += 1  # Comparison operation
        
    return num_cycles, mem_cycles, hw_need


def parse_code(string, expr_type="expr", unrolled=1, loop_iters=1):
    """Parse a code string and return hardware synthesis metrics.
    
    Args:
        string: The code string to parse
        expr_type: Type of expression ("expr", "assign", "augassign", "if", "for", "while", "return")
        unrolled: Unroll factor for loops
        loop_iters: Number of loop iterations
        
    Returns:
        tuple: (cycles, hw_need, memory_cycles) - Hardware synthesis metrics
    """
    # Initialize hardware needs
    hw_need = {key: 0 for key in op2sym_map.keys()}
    hw_need["Regs"] = 0
    memory_cycles = 0
    
    # Convert expression to AST
    try:
        tree = ast.parse(string)
    except:
        return 0, hw_need, memory_cycles
    
    # Process expression based on type
    if expr_type == "assign":
        cycles = latency["store"]
        memory_cycles = latency["store"]
        hw_need["Regs"] += 2  # Source and destination registers
        
    elif expr_type == "augassign":
        cycles = latency["load"] + latency["store"]
        memory_cycles = latency["load"] + latency["store"]
        hw_need["Regs"] += 3  # Source, destination, and temp registers
        
    else:  # Default expr type
        cycles = latency["load"]
        memory_cycles = latency["load"]
        hw_need["Regs"] += 1  # Result register
    
    # Process AST nodes
    for node in ast.walk(tree):
        if isinstance(node, ast.BinOp):
            op_type = type(node.op).__name__
            if op_type in op2sym_map:
                cycles += latency[op_type]
                hw_need[op_type] += 1
                hw_need["Regs"] += 2  # Operand registers
                
        elif isinstance(node, ast.Call):
            cycles += latency["call"]
            memory_cycles += latency["load"]  # Function arguments
            hw_need["Regs"] += len(node.args)  # Argument registers
            
        elif isinstance(node, ast.Compare):
            cycles += latency["compare"]
            for op in node.ops:
                op_type = type(op).__name__
                if op_type in op2sym_map:
                    hw_need[op_type] += 1
            hw_need["Regs"] += 2  # Operand registers
    
    # Scale metrics by unroll factor and loop iterations
    cycles *= unrolled
    memory_cycles *= unrolled
    
    return cycles, hw_need, memory_cycles


def check_and_parse(string, unrolled=1, loop_iters=1):
    """Check and parse a code string for hardware synthesis.
    
    Args:
        string: The code string or AST node to parse
        unrolled: Unroll factor for loops
        loop_iters: Number of loop iterations
        
    Returns:
        tuple: (cycles, hw_need, memory_cycles) - Hardware synthesis metrics
    """
    # Initialize hardware needs
    hw_need = {key: 0 for key in op2sym_map.keys()}
    hw_need["Regs"] = 0
    
    # Convert AST node to source code if needed
    if isinstance(string, ast.AST):
        string = astor.to_source(string)
    
    # Process function calls
    if "def" in string:
        cycles = latency["call"]
        memory_cycles = latency["load"] + latency["store"]  # Function parameters and return value
        hw_need["Regs"] += 4  # Function parameters and return value
        
    elif isinstance(string, ast.Compare):
        # Handle comparisons
        cycles = latency["compare"]
        memory_cycles = 0
        hw_need["Lt"] += 1  # Comparison unit
        hw_need["Regs"] += 2  # Operand registers
        
    else:
        # Default parsing
        cycles, hw_expr, memory_cycles = parse_code(string, "expr", unrolled, loop_iters)
        if isinstance(hw_expr, dict):
            for key in hw_expr:
                hw_need[key] = max(hw_need[key], hw_expr[key])
    
    return cycles, hw_need, memory_cycles

def parse_graph(graph, dse_input=0, dse_given=False, given_bandwidth=1000000, tech_node='45nm'):
    """
    Parse a non-AI workload graph and store the configuration as a hardware representation.
    Supports technology node scaling for power and latency values.
    
    Args:
        graph: The control flow graph to parse
        dse_input: Design space exploration input parameters
        dse_given: Whether DSE parameters are provided
        given_bandwidth: Available memory bandwidth in bytes/sec
        tech_node: Target technology node (default 45nm)
        
    Returns:
        tuple: (cycles, hw_allocated, memory_cfgs) - Hardware synthesis results
    """
    # Initialize global state
    global bw_avail, latency, power, cycles, hw_allocated, hw_utilized, memory_cfgs
    bw_avail = given_bandwidth
    cycles = 0
    total_memory_cycles = 0
    
    # Reset hardware allocation tracking
    hw_allocated = {key: 0 for key in op2sym_map.keys()}
    hw_utilized = {key: 0 for key in op2sym_map.keys()}
    hw_allocated["Regs"] = 0
    hw_utilized["Regs"] = 0

    # Get technology scaling factor
    tech_scale = graph.get_tech_scaling() if hasattr(graph, 'get_tech_scaling') else 1.0

    # Scale latency and power values based on technology node
    scaled_latency = {k: v * tech_scale for k, v in latency.items()}
    scaled_power = {k: [p * tech_scale for p in v] for k, v in power.items()}
    
    # Use scaled values
    latency.update(scaled_latency)
    power.update(scaled_power)

    # Initialize memory state tracking
    memory_cfgs = {}
    mem_state = {}
    variables = {}
    
    # Track loop nesting for better resource allocation
    loop_depth = 0
    max_loop_depth = 0
    total_loop_iters = 1
    matrix_size = 0
    
    # Process each node in the graph
    for node in graph:
        for stmt in node.statements:
            if isinstance(stmt, ast.For):
                loop_depth += 1
                max_loop_depth = max(max_loop_depth, loop_depth)
                
                # Get loop parameters
                loop_iters = 16  # Default size
                unroll_factor = 1
                
                if isinstance(stmt.iter.args[0], ast.Constant):
                    loop_iters = stmt.iter.args[0].value
                    matrix_size = max(matrix_size, loop_iters)
                elif dse_given and "loop1" in dse_input:
                    loop_iters = dse_input["loop1"][0]
                    matrix_size = max(matrix_size, loop_iters)
                    unroll_factor = dse_input["loop1"][1]
                
                total_loop_iters *= loop_iters
                
                # Process loop body with unrolling
                loop_cycles = 0
                loop_memory_cycles = 0
                loop_hw = {key: 0 for key in op2sym_map.keys()}
                loop_hw["Regs"] = 0
                
                for expr in stmt.body:
                    cycles_expr, hw_expr, mem_expr = check_and_parse(
                        expr, 
                        unroll_factor,
                        loop_iters
                    )
                    
                    # Accumulate cycles and resources for this loop iteration
                    loop_cycles += cycles_expr
                    loop_memory_cycles += mem_expr
                    for key in hw_expr:
                        loop_hw[key] = max(loop_hw[key], hw_expr[key])
                
                # Scale loop metrics by iterations and unrolling
                total_loop_cycles = loop_cycles * (loop_iters // unroll_factor)
                total_loop_memory_cycles = loop_memory_cycles * (loop_iters // unroll_factor)
                
                # Update global metrics
                cycles += total_loop_cycles
                total_memory_cycles += total_loop_memory_cycles
                
                # Scale hardware resources based on unrolling and loop depth
                # Use linear scaling with minimum resource requirements
                resource_scale = max(1, unroll_factor * loop_depth)
                for key in loop_hw:
                    if key == "Regs":
                        # Ensure minimum register count for loop variables and data
                        min_regs = 3 * loop_depth + 16  # Loop vars + data buffers
                        loop_hw[key] = max(loop_hw[key], min_regs)
                    elif key in ["Add", "Mult"]:
                        # Ensure minimum arithmetic units based on loop iterations
                        min_units = max(1, loop_hw[key] * unroll_factor // 2)  # Less aggressive sharing
                        loop_hw[key] = max(loop_hw[key], min_units)
                    
                    hw_allocated[key] = max(
                        hw_allocated[key],
                        loop_hw[key] * resource_scale
                    )
                
                loop_depth -= 1
                
            else:
                # Handle non-loop statements
                if isinstance(stmt, ast.Assign):
                    if isinstance(stmt.value, ast.Tuple):
                        if all(isinstance(x, ast.Constant) for x in stmt.value.elts):
                            if isinstance(stmt.targets[0], ast.Name):
                                variables[stmt.targets[0].id] = len(stmt.value.elts)
                    else:
                        cycles_stmt, hw_stmt, mem_stmt = parse_code(astor.to_source(stmt), "assign")
                        cycles += cycles_stmt
                        total_memory_cycles += mem_stmt
                        for key in hw_stmt:
                            hw_allocated[key] = max(hw_allocated[key], hw_stmt[key])
    
    # If using systolic array, adjust resources
    if dse_given and dse_input.get("systolic", False):
        pe_array_size = min(8, dse_input["loop1"][1])  # Up to 8x8 PE array
        
        # Calculate required resources per PE
        mults_per_pe = 4  # Multiple multipliers per PE for better throughput
        adds_per_pe = 4   # Multiple adders per PE for accumulation and forwarding
        regs_per_pe = 8   # More registers for pipelining
        
        # Scale resources by PE array size
        hw_allocated["Mult"] = pe_array_size * pe_array_size * mults_per_pe
        hw_allocated["Add"] = pe_array_size * pe_array_size * adds_per_pe
        hw_allocated["Regs"] = pe_array_size * pe_array_size * regs_per_pe
        
        # Account for control logic
        hw_allocated["Lt"] += pe_array_size * 2  # More comparators for control
        hw_allocated["LtE"] += pe_array_size * 2  # More comparators for bounds
        hw_allocated["Sub"] += pe_array_size * 4  # More address calculations
        
        # Calculate systolic array cycles
        pipeline_depth = 2 * pe_array_size - 1  # Initial fill + drain
        compute_cycles = matrix_size * matrix_size / (pe_array_size * pe_array_size)  # Parallel computation
        memory_cycles = total_memory_cycles / (pe_array_size * 2)  # Memory bandwidth (input + output)
        
        # Total cycles is max of computation and memory access
        cycles = pipeline_depth + max(compute_cycles, memory_cycles)
        
        # Add overhead for wavefront synchronization
        sync_overhead = pe_array_size * 0.1  # 10% overhead per PE row
        cycles *= (1 + sync_overhead)
    else:
        # For basic implementation, ensure minimum resources
        hw_allocated["Regs"] = max(hw_allocated["Regs"], 48)  # Minimum registers for matrix multiply
        hw_allocated["Add"] = max(hw_allocated["Add"], 4)  # More adders for better throughput
        hw_allocated["Mult"] = max(hw_allocated["Mult"], 4)  # More multipliers for better throughput
        
        # Account for sequential execution and memory access
        cycles = (
            total_loop_iters * (latency["Mult"] + latency["Add"]) +  # Computation
            total_memory_cycles  # Memory access
        )
    
    return cycles, hw_allocated, memory_cfgs


def get_params(dfg, area_budget):
    """Adjust parameters to meet area budget

    Args:
        dfg: Data flow graph
        area_budget: Target area constraint
    """
    allocated_area = 0
    while allocated_area < 0.9 * area_budget or allocated_area > 1.2 * area_budget:
        # unroll_params -> modify
        # memory size -> modify
        if area_budget > allocated_area:
            for param in unroll_params.keys():  # Using unroll_params dictionary defined earlier
                # decrease parallelism
                # unroll_params --
                pass
            for mem_cfg in memory_cfgs.keys():  # Using memory_cfgs dictionary defined earlier
                # high registers to sram
                # decreases bandwidth
                # update_memory_cfgs
                pass
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
            mem_list[key] = power[key][0] * value
    return mem_list


def prune_allocator(node=None, func=None):
    """Prune and allocate resources

    Args:
        node: Node to process
        func: Function to allocate

    Returns:
        list: Allocated nodes
    """
    # conflict graph
    # interval graph for registers
    if node and node.operator == "func":
        getall = []
        if func:
            for i in func:
                getall.append(allocate_node(i))
        return getall
    return []


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
