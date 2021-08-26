# from transformations import transform_graph

# from ..scheduling import run_asap


def get_bottlenecks(graph, hw):
    """Finds the Bottlenecks from Timing Gradients (ASAP Mapped)

    Args:
        graph (): 
        algo (): 
        hw (): 

    Returns:
        : 
    """
    bottlenecks = run_asap(hw)
    print(bottlenecks)
    return bottleneckss


def optim_exec_bottlenecks(graph, hw):
    """Optimizes Execution Bottlenecks

    Args:
        graph (): 
        hw (): 

    Returns:
        : 
    """
    for bottleneck in bottlenecks:
        node = lib_match(bottleneck.node())
        if is_tranform(node.type):
            new_node = transform(node.type, node)
        graph.edit(node, new_node, index)
    print(graph, hw)
    return graph


def full(graph, hw):
    graph = transform_graph(graph)
    bottlenecks = get_bottlenecks(graph, hw)
    graph = optim_exec_bottlenecks(graph, hw)
    # print(graph.node[0], graph.node[1])
    # run graph on actual hardware -> write back AST -> code
    # don't run graph on actual hardware -> estimate performance on simulated hardware
    print(perf(graph, hw, mapping="asap"))


def transform_node(type, node):
    pass


# Get AST IR -> transform to a different IR
# what is optimize dfg ? for synthesis ? already done ?
def optimize_dfg():
    pass
