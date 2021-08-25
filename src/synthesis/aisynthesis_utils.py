def complete_functional_config(graph, config, area_constraint=0):
    """
    fn(area constraint, graph)
    analyze graph to get a good initial value of hw description
    config.create()
    gen_systl_fn()
    write as 0_hw.yaml
    """
    config = generate_systolic_array(graph, config)
    config = generate_local_mem(graph, config)
    return config


def generate_local_mem(graph, config):
    """
    fn(config)
    scratchpad config from HW config
    return full_config
    """

    return config


def generate_systolic_array(graph, config):
    """
    systolic array config from mapping efficiency
    return full_config
    """
    total_eff = 0
    min_eff = 1
    total_expense = 0
    for node in graph.nodes:
        total_expense += node.compute_expense
    for i in range(4, 9):
        for j in range(4, 9):
            s_i = 2 ** i
            s_j = 2 ** j
            for node in graph.nodes:
                total_eff += (
                    node.compute_expense
                    * get_efficiency(node, [s_i, s_j])
                    / total_expense
                )
                if total_eff < min_eff:
                    min_i = s_i
                    min_j = s_j
                    min_eff = total_eff
        config["PE_array_size"] = min_i
    return config


def get_efficiency(graph_node, array_size):
    """[Efficiency of node on Systolic Array]

    Args:
        graph_node ([type]): [description]
        array_size ([type]): [description]
    """
    efficiency = 0
    if node.type == "aten::convolution":
        cycles = (N * N * C * R * R * K * K * Co * Ci) / (N * N * B)
        efficiency = cycles / (N * N * B)

    return efficiency


# class ai_graph_manipulations():
#   def __init__(graph):
#     self.graph = graph
#   def smart_topo_sort():
#     # [[a,b],c,d,[e,f,g]]
#     # account_relevant_edges():
#     pass
#   def check_size():
#     # run in parallel
#     pass
#   def dependency_nodes():
#     pass
#   def simplify_edge_mesh():
#     # model internal data movement
#     pass
