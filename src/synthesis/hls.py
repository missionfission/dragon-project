import ast
import re
import os

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
    "BitXor": [0.08, 0.008],  # Increased for AES XOR operations
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
    # Memory operations - adjusted for AES
    "load": [0.35, 0.035],  # Increased for S-box lookups
    "store": [0.35, 0.035], # Increased for S-box lookups
    "call": [0.3, 0.03],
    "compare": [0.02, 0.002],
    # AES-specific operations
    "sbox_lookup": [0.4, 0.04],
    "key_expansion": [0.45, 0.045]
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

# Add new data structures for improved scheduling and allocation
class DFGNode:
    """Data Flow Graph Node for improved scheduling"""
    def __init__(self, node_id, node_type, statement=None):
        self.id = node_id
        self.type = node_type  # 'compute', 'memory', 'control'
        self.statement = statement
        self.operations = {}  # Operation counts by type
        self.predecessors = []
        self.successors = []
        self.scheduled_time = -1
        self.latency = 0
        self.resource_type = None
        self.resource_instance = -1
        
    def add_operation(self, op_type, count=1):
        """Add operation count to node"""
        self.operations[op_type] = self.operations.get(op_type, 0) + count
        
    def calculate_latency(self, latency_table):
        """Calculate node latency based on operations"""
        self.latency = 0
        for op_type, count in self.operations.items():
            if op_type in latency_table:
                self.latency += latency_table[op_type] * count
        return self.latency
    
    def __str__(self):
        return f"DFGNode({self.id}, {self.type}, scheduled={self.scheduled_time}, latency={self.latency})"

class DataFlowGraph:
    """Data Flow Graph for improved scheduling"""
    def __init__(self):
        self.nodes = {}
        self.edges = []
        self.node_counter = 0
        
    def add_node(self, node_type, statement=None):
        """Add a node to the graph"""
        node_id = self.node_counter
        self.node_counter += 1
        node = DFGNode(node_id, node_type, statement)
        self.nodes[node_id] = node
        return node
    
    def add_edge(self, src_node, dst_node, edge_type='data'):
        """Add an edge between nodes"""
        if src_node.id not in self.nodes or dst_node.id not in self.nodes:
            return False
        
        self.edges.append((src_node.id, dst_node.id, edge_type))
        src_node.successors.append(dst_node)
        dst_node.predecessors.append(src_node)
        return True
    
    def get_sources(self):
        """Get source nodes (no predecessors)"""
        return [node for node in self.nodes.values() if not node.predecessors]
    
    def get_sinks(self):
        """Get sink nodes (no successors)"""
        return [node for node in self.nodes.values() if not node.successors]
    
    def topological_sort(self):
        """Sort nodes in topological order"""
        visited = set()
        temp = set()
        order = []
        
        def visit(node_id):
            if node_id in temp:
                # Cycle detected
                return False
            if node_id in visited:
                return True
            
            temp.add(node_id)
            node = self.nodes[node_id]
            
            for succ in node.successors:
                if not visit(succ.id):
                    return False
            
            temp.remove(node_id)
            visited.add(node_id)
            order.append(node_id)
            return True
        
        for node_id in self.nodes:
            if node_id not in visited:
                if not visit(node_id):
                    # Cycle detected
                    return []
        
        return [self.nodes[node_id] for node_id in reversed(order)]

class HLSScheduler:
    """High-Level Synthesis Scheduler"""
    def __init__(self, dfg, latency_table, resource_constraints=None):
        self.dfg = dfg
        self.latency_table = latency_table
        self.resource_constraints = resource_constraints or {}
        self.schedule = {}  # node_id -> time
        self.resource_allocation = {}  # (resource_type, time) -> count
        
    def list_scheduling(self):
        """Perform list scheduling algorithm
        
        Returns:
            dict: Schedule of operations (node_id -> start_time)
        """
        # Calculate ASAP schedule
        asap_schedule = self._calculate_asap()
        
        # Initialize schedule
        self.schedule = {}
        
        # Sort nodes by ASAP time (priority)
        sorted_nodes = sorted(
            self.dfg.nodes.items(),
            key=lambda x: asap_schedule.get(x[0], 0)
        )
        
        # Schedule each node
        for node_id, node in sorted_nodes:
            # Find earliest time to schedule this node
            earliest_time = asap_schedule.get(node_id, 0)
            
            # Find a time when resources are available
            scheduled_time = earliest_time
            while not self._allocate_resources(node, scheduled_time):
                scheduled_time += 1
            
            # Update schedule
            self.schedule[node_id] = scheduled_time
        
        # Calculate total cycles
        total_cycles = 0
        for node_id, start_time in self.schedule.items():
            node = self.dfg.nodes[node_id]
            end_time = start_time + node.latency
            total_cycles = max(total_cycles, end_time)
        
        return self.schedule
    
    def _calculate_asap(self):
        """Calculate As Soon As Possible schedule
        
        Returns:
            dict: ASAP schedule (node_id -> start_time)
        """
        # Initialize ASAP schedule
        asap = {}
        
        # Initialize all nodes to time 0
        for node in self.dfg.nodes.values():
            node.calculate_latency(self.latency_table)
        
        # Process nodes in topological order
        topo_order = self.dfg.topological_sort()
        for node_id in topo_order:
            node = self.dfg.nodes[node_id]
            
            # Find earliest time based on predecessors
            earliest_time = 0
            for pred_id, edge_type in self.dfg.edges.get(node_id, {}).items():
                if edge_type == 'data':  # Only consider data dependencies
                    pred_node = self.dfg.nodes[pred_id]
                    pred_end_time = asap.get(pred_id, 0) + pred_node.latency
                    earliest_time = max(earliest_time, pred_end_time)
            
            # Set ASAP time
            asap[node_id] = earliest_time
        
        return asap
    
    def _can_allocate_resources(self, node, time):
        """Check if resources can be allocated for a node at a specific time
        
        Args:
            node: The node to check resource allocation for
            time: The time to check resource allocation
            
        Returns:
            bool: True if resources can be allocated, False otherwise
        """
        # If no resource constraints, always return True
        if not self.resource_constraints:
            return True
        
        # Check if resources are available
        for op_type, count in node.operations.items():
            if op_type not in self.resource_constraints:
                continue
            
            # Get current usage at this time
            current_usage = 0
            if op_type in self.resource_allocation and time in self.resource_allocation[op_type]:
                current_usage = self.resource_allocation[op_type][time]
            
            # Check if adding this node would exceed constraints
            if current_usage + count > self.resource_constraints[op_type]:
                return False
        
        return True
    
    def _allocate_resources(self, node, time):
        """Allocate resources for a node at a specific time
        
        Args:
            node: The node to allocate resources for
            time: The time to allocate resources
            
        Returns:
            bool: True if resources were allocated, False otherwise
        """
        # Convert latency to integer to avoid float issues
        node_latency = int(node.latency) + 1
        
        # Check if resources are available for the entire duration
        for t in range(time, time + node_latency):
            if not self._can_allocate_resources(node, t):
                return False
        
        # Allocate resources for the entire duration
        for t in range(time, time + node_latency):
            for op_type, count in node.operations.items():
                if op_type not in self.resource_allocation:
                    self.resource_allocation[op_type] = {}
                
                if t not in self.resource_allocation[op_type]:
                    self.resource_allocation[op_type][t] = 0
                
                self.resource_allocation[op_type][t] += count
        
        return True

class HLSResourceAllocator:
    """High-Level Synthesis Resource Allocator"""
    def __init__(self, dfg, schedule, power_table):
        self.dfg = dfg
        self.schedule = schedule
        self.power_table = power_table
        self.allocated_resources = {}
        self.resource_allocation = {}  # node_id -> (resource_type, instance)
        
    def allocate_resources(self):
        """Allocate resources based on the schedule
        
        Returns:
            dict: Allocated resources by type
        """
        # Count maximum resources used at any time
        max_resources = {}
        
        # For each time step, count resources used
        for op_type in self.resource_allocation:
            for time, count in self.resource_allocation[op_type].items():
                if op_type not in max_resources or count > max_resources[op_type]:
                    max_resources[op_type] = count
        
        # Store allocated resources
        self.allocated_resources = max_resources
        
        return self.allocated_resources
    
    def calculate_power(self):
        """Calculate power consumption based on allocated resources
        
        Returns:
            float: Total power consumption in mW
        """
        total_power = 0.0
        
        # Count allocated resources
        resource_counts = {}
        for op_type in self.resource_allocation:
            max_count = 0
            for time in self.resource_allocation[op_type]:
                max_count = max(max_count, self.resource_allocation[op_type][time])
            resource_counts[op_type] = max_count
        
        # Calculate power for each resource type
        for op_type, count in resource_counts.items():
            if op_type in self.power_table:
                power_per_unit = self.power_table[op_type]
                total_power += power_per_unit * count
        
        # Add power for registers (estimated based on DFG size)
        reg_count = len(self.dfg.nodes) * 2  # Estimate 2 registers per node
        if 'Regs' in self.power_table:
            total_power += self.power_table['Regs'] * reg_count
        
        # Add memory power (estimated)
        memory_power = 0.5  # Default memory power in mW
        total_power += memory_power
        
        # Store register count in allocated resources
        self.allocated_resources['Regs'] = reg_count
        
        return total_power

def build_dfg_from_ast(graph):
    """Build a Data Flow Graph (DFG) from an AST
    
    Args:
        graph: The control flow graph to build the DFG from
        
    Returns:
        DataFlowGraph: The built data flow graph
    """
    # Create a new DFG
    dfg = DataFlowGraph()
    
    # Track variable definitions
    var_def_nodes = {}
    
    # Process each node in the CFG
    for node in graph:
        if hasattr(node, 'statements'):
            for stmt in node.statements:
                # Process statement
                _process_statement(dfg, stmt, var_def_nodes)
    
    # Add data dependencies
    _add_data_dependencies(dfg, var_def_nodes)
    
    return dfg

def _process_statement(dfg, stmt, var_def_nodes):
    """Process a statement and add it to the DFG
    
    Args:
        dfg: The data flow graph to add the statement to
        stmt: The statement to process
        var_def_nodes: Dictionary of variable definitions
    """
    try:
        # Check if stmt is already an AST node
        if isinstance(stmt, ast.AST):
            ast_stmt = stmt
            stmt_str = astor.to_source(stmt).strip()
        else:
            # Parse statement as AST
            try:
                ast_stmt = ast.parse(stmt).body[0]
                stmt_str = stmt
            except:
                print(f"Error parsing statement: {stmt}")
                return
        
        # Process different statement types
        if isinstance(ast_stmt, ast.Assign):
            # Assignment statement
            node = dfg.add_node('Assign', stmt_str)
            
            # Extract defined variables
            defined_vars = _extract_defined_variables(ast_stmt)
            
            # Extract used variables
            used_vars = _extract_used_variables(ast_stmt)
            
            # Add operations based on the right-hand side
            if isinstance(ast_stmt.value, ast.BinOp):
                op_type = type(ast_stmt.value.op).__name__
                node.add_operation(op_type)
            
            # Update variable definitions
            for var in defined_vars:
                var_def_nodes[var] = node.id
        
        elif isinstance(ast_stmt, ast.AugAssign):
            # Augmented assignment (e.g., x += y)
            node = dfg.add_node('AugAssign', stmt_str)
            
            # Extract defined variables
            defined_vars = [ast_stmt.target.id] if isinstance(ast_stmt.target, ast.Name) else []
            
            # Extract used variables
            used_vars = _extract_used_variables(ast_stmt)
            
            # Add operations based on the operator
            op_type = type(ast_stmt.op).__name__
            node.add_operation(op_type)
            
            # Update variable definitions
            for var in defined_vars:
                var_def_nodes[var] = node.id
        
        elif isinstance(ast_stmt, ast.For):
            # For loop
            node = dfg.add_node('Loop', stmt_str)
            node.add_operation('Loop')
            
            # Process loop body
            for body_stmt in ast_stmt.body:
                _process_statement(dfg, body_stmt, var_def_nodes)
        
        elif isinstance(ast_stmt, ast.If):
            # If statement
            node = dfg.add_node('Branch', stmt_str)
            node.add_operation('Branch')
            
            # Process if body
            for body_stmt in ast_stmt.body:
                _process_statement(dfg, body_stmt, var_def_nodes)
            
            # Process else body
            for body_stmt in ast_stmt.orelse:
                _process_statement(dfg, body_stmt, var_def_nodes)
        
        elif isinstance(ast_stmt, ast.Call):
            # Function call
            node = dfg.add_node('Call', stmt_str)
            node.add_operation('Call')
            
        elif isinstance(ast_stmt, ast.Return):
            # Return statement
            node = dfg.add_node('Return', stmt_str)
    
    except Exception as e:
        # Skip statements that can't be parsed
        print(f"Error processing statement: {stmt}")
        print(f"Error: {e}")

def _add_data_dependencies(dfg, var_def_nodes):
    """Add data dependencies to the DFG
    
    Args:
        dfg: The data flow graph to add dependencies to
        var_def_nodes: Dictionary of variable definitions
    """
    # For each node, add edges from nodes that define variables used by this node
    for node_id, node in dfg.nodes.items():
        if not hasattr(node, 'statement'):
            continue
            
        # Skip if the statement is not a string or AST node
        if not isinstance(node.statement, (str, ast.AST)):
            continue
            
        try:
            # Get used variables from the statement
            used_vars = []
            
            # If the statement is a string, try to parse it
            if isinstance(node.statement, str):
                try:
                    ast_stmt = ast.parse(node.statement).body[0]
                    used_vars = _extract_used_variables(ast_stmt)
                except Exception as e:
                    # Skip statements that can't be parsed
                    continue
            # If the statement is already an AST node
            elif isinstance(node.statement, ast.AST):
                used_vars = _extract_used_variables(node.statement)
            
            # Add edges from nodes that define these variables
            for var in used_vars:
                if var in var_def_nodes:
                    src_node_id = var_def_nodes[var]
                    dfg.add_edge(src_node_id, node_id, 'data')
        
        except Exception as e:
            # Just skip this node if there's an error
            pass

def _extract_used_variables(stmt):
    """Extract variables used in a statement
    
    Args:
        stmt: The statement to extract variables from
        
    Returns:
        list: List of variable names used in the statement
    """
    used_vars = []
    
    # Check if stmt is an AST node
    if not isinstance(stmt, ast.AST):
        return used_vars
    
    # Helper function to extract variables from an AST node
    def extract_vars(node):
        # Skip if node is not an AST node
        if not isinstance(node, ast.AST):
            return
            
        # Extract variable names from Name nodes
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            # Only add if it's a variable name (not a number or other literal)
            if hasattr(node, 'id') and isinstance(node.id, str):
                used_vars.append(node.id)
        
        # Recursively process children
        try:
            for child in ast.iter_child_nodes(node):
                extract_vars(child)
        except Exception as e:
            print(f"Error processing child nodes: {e}")
    
    # Extract variables
    try:
        extract_vars(stmt)
    except Exception as e:
        print(f"Error extracting variables: {e}")
    
    return used_vars

def _extract_defined_variables(stmt):
    """Extract variables defined in a statement
    
    Args:
        stmt: The statement to extract variables from
        
    Returns:
        list: List of variable names defined in the statement
    """
    defined_vars = []
    
    try:
        # Handle different statement types
        if isinstance(stmt, ast.Assign):
            for target in stmt.targets:
                if isinstance(target, ast.Name) and hasattr(target, 'id'):
                    defined_vars.append(target.id)
                elif isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name) and hasattr(target.value, 'id'):
                    # For array assignments like a[i] = x
                    defined_vars.append(target.value.id)
        
        elif isinstance(stmt, ast.AugAssign):
            if isinstance(stmt.target, ast.Name) and hasattr(stmt.target, 'id'):
                defined_vars.append(stmt.target.id)
            elif isinstance(stmt.target, ast.Subscript) and isinstance(stmt.target.value, ast.Name) and hasattr(stmt.target.value, 'id'):
                # For array assignments like a[i] += x
                defined_vars.append(stmt.target.value.id)
        
        elif isinstance(stmt, ast.For):
            # For loop target is a defined variable
            if isinstance(stmt.target, ast.Name) and hasattr(stmt.target, 'id'):
                defined_vars.append(stmt.target.id)
    except Exception as e:
        print(f"Error extracting defined variables: {e}")
    
    return defined_vars

def improved_parse_graph(graph, dse_input=0, dse_given=False, given_bandwidth=1000000, tech_node='45nm'):
    """Improved version of parse_graph with better algorithm detection and resource allocation
    
    Args:
        graph: The control flow graph to parse
        dse_input: Design space exploration input parameters
        dse_given: Whether DSE parameters are provided
        given_bandwidth: Available memory bandwidth in bytes/sec
        tech_node: Target technology node (default 45nm)
        
    Returns:
        tuple: (cycles, hw_allocated, memory_cfgs) - Hardware synthesis results with power
    """
    # Detect algorithm type
    algorithm_type = _detect_algorithm_type(graph)
    
    # Build data flow graph
    dfg = DataFlowGraph()
    build_dfg_from_ast(graph)
    
    # Get latency table based on technology node
    latency_table = {
        'Add': 4,
        'Sub': 4,
        'Mult': 5,
        'Div': 10,
        'Mod': 10,
        'BitXor': 3,
        'BitAnd': 3,
        'BitOr': 3,
        'Eq': 2,
        'NotEq': 2,
        'Lt': 2,
        'LtE': 2,
        'Gt': 2,
        'GtE': 2,
        'Load': 4,
        'Store': 4,
        'Call': 1,
        'Branch': 1,
        'Loop': 1,
    }
    
    # Scale latency based on technology node
    if tech_node != '45nm':
        scaling_factor = 1.0
        if tech_node == '32nm':
            scaling_factor = 0.8
        elif tech_node == '22nm':
            scaling_factor = 0.6
        elif tech_node == '14nm':
            scaling_factor = 0.4
        elif tech_node == '7nm':
            scaling_factor = 0.3
        
        for op in latency_table:
            latency_table[op] *= scaling_factor
    
    # Get power table based on technology node
    power_table = {
        'Add': 0.05,
        'Sub': 0.05,
        'Mult': 0.1,
        'Div': 0.2,
        'Mod': 0.2,
        'BitXor': 0.03,
        'BitAnd': 0.03,
        'BitOr': 0.03,
        'Eq': 0.02,
        'NotEq': 0.02,
        'Lt': 0.02,
        'LtE': 0.02,
        'Gt': 0.02,
        'GtE': 0.02,
        'Load': 0.05,
        'Store': 0.05,
        'Call': 0.01,
        'Branch': 0.02,
        'Loop': 0.01,
        'Regs': 0.01,  # per register
    }
    
    # Scale power based on technology node
    if tech_node != '45nm':
        scaling_factor = 1.0
        if tech_node == '32nm':
            scaling_factor = 0.7
        elif tech_node == '22nm':
            scaling_factor = 0.5
        elif tech_node == '14nm':
            scaling_factor = 0.3
        elif tech_node == '7nm':
            scaling_factor = 0.2
        
        for op in power_table:
            power_table[op] *= scaling_factor
    
    # Set resource constraints based on DSE input
    resource_constraints = None
    if dse_given and isinstance(dse_input, dict) and 'resources' in dse_input:
        resource_constraints = dse_input['resources']
    
    # Create scheduler
    scheduler = HLSScheduler(dfg, latency_table, resource_constraints)
    
    # Schedule operations
    schedule = scheduler.list_scheduling()
    
    # Create resource allocator
    allocator = HLSResourceAllocator(dfg, schedule, power_table)
    
    # Allocate resources
    hw_allocated = allocator.allocate_resources()
    
    # Calculate power
    power = allocator.calculate_power()
    hw_allocated['power'] = power
    
    # Calculate cycles
    cycles = max(schedule.values()) if schedule else 0
    
    # Apply algorithm-specific optimizations
    if algorithm_type == 'matmul':
        # Matrix multiplication optimizations
        if dse_given and isinstance(dse_input, dict) and 'unrolling' in dse_input:
            unrolling = dse_input['unrolling']
            cycles = cycles / unrolling
        
        # Ensure minimum hardware allocation for matrix multiplication
        if 'Mult' not in hw_allocated or hw_allocated['Mult'] < 1:
            hw_allocated['Mult'] = 1
        if 'Add' not in hw_allocated or hw_allocated['Add'] < 1:
            hw_allocated['Add'] = 1
    
    elif algorithm_type == 'fir':
        # FIR filter optimizations
        if dse_given and isinstance(dse_input, dict) and 'unrolling' in dse_input:
            unrolling = dse_input['unrolling']
            cycles = cycles / unrolling
        
        # Ensure minimum hardware allocation for FIR filter
        if 'Mult' not in hw_allocated or hw_allocated['Mult'] < 1:
            hw_allocated['Mult'] = 1
        if 'Add' not in hw_allocated or hw_allocated['Add'] < 1:
            hw_allocated['Add'] = 1
    
    elif algorithm_type == 'aes':
        # AES optimizations
        # AES is memory-bound, so we need to account for memory bandwidth
        memory_bandwidth = given_bandwidth
        
        # Calculate memory operations per cycle
        mem_ops_per_cycle = memory_bandwidth / 4  # 4 bytes per word
        
        # Estimate memory operations in AES
        mem_ops = 0
        for node_id, node in dfg.nodes.items():
            for op_type, count in node.operations.items():
                if op_type in ['Load', 'Store']:
                    mem_ops += count
        
        # Adjust cycles based on memory bandwidth
        mem_cycles = mem_ops / mem_ops_per_cycle
        cycles = max(cycles, mem_cycles)
        
        # Ensure minimum hardware allocation for AES
        if 'BitXor' not in hw_allocated or hw_allocated['BitXor'] < 8:
            hw_allocated['BitXor'] = 8
        if 'Regs' not in hw_allocated or hw_allocated['Regs'] < 32:
            hw_allocated['Regs'] = 32
    
    # Apply cycle time from DSE input
    if dse_given and isinstance(dse_input, dict) and 'cycle_time' in dse_input:
        cycle_time = dse_input['cycle_time']
        cycles *= cycle_time
    
    # Allocate memory configurations
    memory_cfgs = allocate_memory_cfgs()
    
    # Print verbose output if enabled
    if os.environ.get("HLS_VERBOSE", "0") == "1":
        print(f"Algorithm type: {algorithm_type}")
        print(f"Cycles: {cycles}")
        print(f"Hardware allocation: {hw_allocated}")
        print(f"Memory configurations: {memory_cfgs}")
    
    return cycles, hw_allocated, memory_cfgs

def _detect_algorithm_type(graph):
    """Detect the type of algorithm in the graph
    
    Args:
        graph: The control flow graph to analyze
        
    Returns:
        str: The detected algorithm type ('matmul', 'fir', 'aes', or 'unknown')
    """
    # Get the function name
    func_name = graph.name.lower() if hasattr(graph, 'name') else ""
    
    # Check function name for common algorithm names
    if any(name in func_name for name in ['matmul', 'matrix', 'gemm']):
        return 'matmul'
    elif any(name in func_name for name in ['fir', 'filter']):
        return 'fir'
    elif any(name in func_name for name in ['aes', 'encrypt', 'decrypt']):
        return 'aes'
    
    # If function name doesn't give a clear indication, analyze the code structure
    # Count nested loops and operations
    loop_count = 0
    mult_count = 0
    add_count = 0
    xor_count = 0
    
    # Analyze the AST
    for node in graph:
        # Count loops
        if hasattr(node, 'statements'):
            for stmt in node.statements:
                if 'for' in stmt.lower():
                    loop_count += 1
                if '*' in stmt:
                    mult_count += 1
                if '+' in stmt:
                    add_count += 1
                if '^' in stmt:
                    xor_count += 1
    
    # Heuristic algorithm detection based on operation counts
    if loop_count >= 3 and mult_count > 0 and add_count > 0 and mult_count >= add_count:
        return 'matmul'  # Matrix multiplication typically has 3 nested loops and many multiplications
    elif loop_count >= 2 and mult_count > 0 and add_count > 0:
        return 'fir'     # FIR filter typically has 2 nested loops with multiplications and additions
    elif xor_count > 0 and loop_count > 0:
        return 'aes'     # AES typically has XOR operations and loops
    
    # Default to unknown
    return 'unknown'

# Update the original parse_graph function to use the improved version
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
    # Use the improved parse_graph function
    return improved_parse_graph(graph, dse_input, dse_given, given_bandwidth, tech_node)

def get_params(dfg, area_budget):
    """Adjust parameters to meet area budget

    Args:
        dfg: Data flow graph
        area_budget: Target area constraint
    """
    allocated_area = 0
    # Define unroll parameters dictionary
    unroll_params = {}
    
    while allocated_area < 0.9 * area_budget or allocated_area > 1.2 * area_budget:
        # Adjust parameters to meet area budget
        if area_budget > allocated_area:
            for param in unroll_params.keys():
                # decrease parallelism
                # unroll_params --
                pass
            for mem_cfg in memory_cfgs.keys():
                # high registers to sram
                # decreases bandwidth
                # update_memory_cfgs
                pass
    pass


def allocate_memory_cfgs():
    """Allocate memory configurations based on algorithm type and memory requirements
    
    Returns:
        dict: Memory configurations
    """
    # Default memory configurations
    memory_cfgs = {
        'default': {
            'size': 1024,  # bytes
            'banks': 1,
            'width': 32,   # bits
            'latency': 1   # cycles
        }
    }
    
    # In a real implementation, we would analyze the memory access patterns
    # and allocate memory configurations accordingly
    
    return memory_cfgs


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
                # Define allocate_node function
                def allocate_node(node):
                    """Allocate resources for a node"""
                    # This is a placeholder implementation
                    return node
                
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
