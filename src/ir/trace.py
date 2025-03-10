import warnings
import numpy as np
import torch
import torch.fx

from ir.flatten import Flatten, flatten
from ir.graph import Graph
from ir.handlers import handlers
from ir.node import Node
from ir.variable import Variable

__all__ = ["trace"]


def get_backprop_memory(nodes):
    total_backprop_mem = 0
    for i, node in enumerate(nodes):
        total_backprop_mem += node.in_edge_mem + node.out_edge_mem + node.weights

    return total_backprop_mem


def trace(model, args=(), kwargs=None):
    assert kwargs is None, (
        "Keyword arguments are not supported for now. "
        "Please use positional arguments instead!"
    )

    # Use torch.fx to capture the graph
    # flattened_model = Flatten(model)
    graph_module = torch.fx.symbolic_trace(model)
    fx_graph = graph_module.graph

    # Create variables dictionary to track inputs/outputs
    variables = dict()
    
    # Helper to get shape and dtype
    def get_tensor_meta(node):
        if hasattr(node, 'meta') and 'tensor_meta' in node.meta:
            meta = node.meta['tensor_meta']
            return Variable(
                name=node.name,
                dtype=meta.dtype,
                shape=meta.shape
            )
        return Variable(name=node.name, dtype=str(node.type) if hasattr(node, 'type') else None)

    # Track all nodes and build dependency graph
    nodes = []
    node_map = {}  # Map from fx node to our Node
    
    # First pass: Create all nodes
    for fx_node in fx_graph.nodes:
        if fx_node.op == 'placeholder':
            var = get_tensor_meta(fx_node)
            variables[fx_node] = var
            continue
            
        if fx_node.op == 'output':
            continue

        # Create our IR node
        node = Node(
            operator=fx_node.op,
            attributes={} if not hasattr(fx_node, 'kwargs') else fx_node.kwargs,
            inputs=[],  # Will be populated in second pass
            outputs=[get_tensor_meta(fx_node)],
            scope=fx_node.name
        )
        
        # Process the node through handlers
        for operators, func in handlers:
            if isinstance(operators, str):
                operators = [operators]
            if node.operator in operators:
                if func is not None:
                    node.compute_expense, node.weights, _ = func(node)
                    node.in_edge_mem = 0  # Will be updated in second pass
                    node.out_edge_mem = np.prod(node.outputs[0].shape) if node.outputs[0].shape else 0
                    node.mem_util = node.weights + node.out_edge_mem
                    if node.compute_expense > 0:
                        nodes.append(node)
                        node_map[fx_node] = node

    # Second pass: Connect inputs
    for i, node in enumerate(nodes):
        fx_node = next(k for k, v in node_map.items() if v == node)
        
        # Connect inputs
        for arg in fx_node.args:
            if isinstance(arg, torch.fx.Node):
                if arg in node_map:  # Connect to previous op node
                    prev_node = node_map[arg]
                    node.inputs.append(prev_node.outputs[0])
                    node.prev.append(prev_node)
                    prev_node.next.append(node)
                    node.in_edge_mem += np.prod(prev_node.outputs[0].shape) if prev_node.outputs[0].shape else 0
                elif arg.op == 'placeholder':  # Connect to input
                    node.inputs.append(variables[arg])
                    node.in_edge_mem += np.prod(variables[arg].shape) if variables[arg].shape else 0

    # Perform topological sort
    sorted_nodes = []
    visited = set()
    temp_mark = set()

    def visit(n):
        if n in temp_mark:
            raise ValueError("Graph has cycles!")
        if n not in visited:
            temp_mark.add(n)
            for next_node in n.next:
                visit(next_node)
            temp_mark.remove(n)
            visited.add(n)
            sorted_nodes.insert(0, n)

    # Start DFS from nodes with no predecessors
    start_nodes = [n for n in nodes if not n.prev]
    for node in start_nodes:
        visit(node)

    # Create the final graph
    graph = Graph(
        name=model.__class__.__module__ + "." + model.__class__.__name__,
        variables=[v for v in variables.values()],
        inputs=[v for k, v in variables.items() if k.op == 'placeholder'],
        outputs=[n.outputs[0] for n in nodes if not n.next],
        nodes=sorted_nodes,
        fx_graph=fx_graph  # Pass the FX graph
    )

    return graph
