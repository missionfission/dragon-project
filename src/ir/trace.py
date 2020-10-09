import warnings

import numpy as np
import torch
import torch.jit

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

    with warnings.catch_warnings(record=True):
        graph, _ = torch.jit._get_trace_graph(Flatten(model), args, kwargs)

    variables = dict()
    for x in graph.nodes():
        for v in list(x.inputs()) + list(x.outputs()):
            if "tensor" in v.type().kind().lower():
                variables[v] = Variable(
                    name=v.debugName(),
                    dtype=v.type().scalarType(),
                    shape=v.type().sizes(),
                )
            else:
                variables[v] = Variable(name=v.debugName(), dtype=str(v.type()),)

    nodes = []
    for x in graph.nodes():
        node = Node(
            operator=x.kind(),
            attributes={s: getattr(x, x.kindOf(s))(s) for s in x.attributeNames()},
            inputs=[variables[v] for v in x.inputs() if v in variables],
            outputs=[variables[v] for v in x.outputs() if v in variables],
            scope=x.scopeName().replace("Flatten/", "", 1).replace("Flatten", "", 1),
        )
        for operators, func in handlers:
            if isinstance(operators, str):
                operators = [operators]
            if node.operator in operators:
                if func is not None:
                    # TODO Merge Small Node
                    # read access are weight read access
                    node.compute_expense, node.weights, _ = func(node)
                    # print(node.weights)
                    # if not isinstance(node.weights, int):
                    #     if len(node.weights) > 1:
                    #         node.weights = node.weights[0]
                    node.in_edge_mem = np.prod(node.inputs[0].shape)
                    node.out_edge_mem = np.prod(node.outputs[0].shape)
                    node.mem_util = node.weights + node.out_edge_mem
                    node.read_access = node.weights + node.in_edge_mem
                    # print(
                    # "inputs",
                    # node.inputs[0].shape,
                    # "outputs",
                    # node.outputs[0].shape,
                    # "weights",
                    # node.weights,
                    # )
                    # if not isinstance(node.mem_util, np.int64):
                    #     if (node.mem_util).shape[0] > 1:
                    #         node.mem_util = node.mem_util[0]
                    if node.compute_expense > 0:
                        nodes.append(node)

    for i, node in enumerate(nodes):
        if i < len(nodes) - 1:
            node.next = nodes[i + 1]
        # print(node.next, node.mem_util)
    graph = Graph(
        name=model.__class__.__module__ + "." + model.__class__.__name__,
        variables=[v for v in variables.values()],
        inputs=[variables[v] for v in graph.inputs() if v in variables],
        outputs=[variables[v] for v in graph.outputs() if v in variables],
        nodes=nodes,
    )
    return graph
