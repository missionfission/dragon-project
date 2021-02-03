import ast
import warnings

import numpy as np
import torch
import torch.jit

from ir.ddfg import DDFG
from ir.ddfg_handlers import ddfg_handlers
from ir.ddfg_node import DDFG_Node
from ir.variable import Variable
from staticfg import CFGBuilder

__all__ = ["ddfg_trace"]


def trace(model, args=(), kwargs=None):
    assert kwargs is None, (
        "Keyword arguments are not supported for now. "
        "Please use positional arguments instead!"
    )

    # Get Trace from a C++ or a python program
    # Invoke trace script

    cfg = CFGBuilder().build_from_file(filename)
    cfg.build_visual("exampleCFG", "pdf", show=False)
    # print(cfg)
    variables = dict()

    for node in cfg:
        for i in node.statements:
            print(ast.dump(i))

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
        node = DDFG_Node(
            operator=x.kind(),
            attributes={s: getattr(x, x.kindOf(s))(s) for s in x.attributeNames()},
            inputs=[variables[v] for v in x.inputs() if v in variables],
            outputs=[variables[v] for v in x.outputs() if v in variables],
            scope=x.scopeName().replace("Flatten/", "", 1).replace("Flatten", "", 1),
        )
        for operators, func in ddfg_handlers:
            if isinstance(operators, str):
                operators = [operators]
            if node.operator in operators:
                if func is not None:
                    # TODO Merge Small Node
                    # TODO Loop merging
                    # TODO Control Statements Merging
                    # read access are weight read access
                    node.compute_expense, node.static_inputs, _ = func(node)
                    # print(node.weights)
                    # if not isinstance(node.weights, int):
                    #     if len(node.weights) > 1:
                    #         node.weights = node.weights[0]
                    node.in_edge_mem = np.prod(
                        node.inputs[0].shape
                    )  # if many previous nodes add them all
                    node.out_edge_mem = np.prod(node.outputs[0].shape)
                    node.mem_util = node.static_inputs + node.out_edge_mem
                    # print("inputs", node.inputs[0].shape, "outputs", node.outputs[0].shape,"weights", node.weights)
                    # if not isinstance(node.mem_util, np.int64):
                    #     if (node.mem_util).shape[0] > 1:
                    #         node.mem_util = node.mem_util[0]

    for i, node in enumerate(nodes):
        if i < len(nodes) - 1:
            node.next = nodes[i + 1]
        # TODO add node prev
        # print(node.next, node.mem_util)
    graph = DDFG(
        name=model.__class__.__module__ + "." + model.__class__.__name__,
        variables=[v for v in variables.values()],
        inputs=[variables[v] for v in graph.inputs() if v in variables],
        outputs=[variables[v] for v in graph.outputs() if v in variables],
        nodes=nodes,
    )
    return graph
