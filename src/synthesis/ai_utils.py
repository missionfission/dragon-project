"""Utilies for Generating and Optimizing Hardware Architectures for AI Workloads
"""


def get_reuse(node):
    """Get Reuse Possible for Conv and Matmul nodes

    Args:
        node (): 
    """
    # for node.type in conv2d
    #
    pass


def complete_functional_config(graph, config, area_constraint=0):
    """
    Analyze Workload to create an initial hardware configuration that satisfies the area constraints -> written in file "iters/0_hw.yaml",
    This will be updated upon interations in the backward_pass_design
    """
    config = generate_systolic_array(graph, config)
    config = generate_local_mem(graph, config)
    return config


def generate_local_mem(graph, config):
    """ Create Scratchpad Memory Config from HW config
    Args:
        graph (): 
        config (): 
    """
    return config


def generate_systolic_array(graph, config):
    """
    Best Systolic Array Sizing for the Entire Workload by Evaluating Mapping Efficiency at Different Sizes

    Args:
        graph (): 
        config (): 
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
    """
    Efficiency of Mapping a node on Systolic Array of Size Array_Size [s_i, s_j]
    Args:
        graph_node (): 
        array_size (): 
    """
    efficiency = 0
    if node.type == "aten::convolution":
        cycles = (N * N * C * R * R * K * K * Co * Ci) / (N * N * B)
        efficiency = cycles / (N * N * B)

    return efficiency

def generate_loop_blocking_fast():
    for ti, to, tb, orders in itertools.product(
        Util.factorize(nested_loop_desc.loopcnt_ifm(), 3),
        Util.factorize(nested_loop_desc.loopcnt_ofm(), 3),
        Util.factorize(nested_loop_desc.loopcnt_bat(), 3),
        itertools.product(
            [None],
            itertools.permutations((de.IFM, de.OFM)),
            [None],
            itertools.permutations((de.IFM, de.OFM)),
        ),
    ):

        if layer_lb is not None:
            if layer_lb["ti"] is not None and ti != layer_lb["ti"]:
                continue
            if layer_lb["to"] is not None and to != layer_lb["to"]:
                continue
            if layer_lb["tb"] is not None and tb != layer_lb["tb"]:
                continue
            if layer_lb["orders"] is not None and orders != layer_lb["orders"]:
                continue

        yield cost_loopblocking_gbuf_regf(
            ti,
            to,
            tb,
            orders,
            resource=resource,
            cost=cost,
            nested_loop_desc=nested_loop_desc,
            options=options,
        )

    ti = np.array(tifm)
    to = np.array(tofm)
    tb = np.array(tbat)

    tip = np.prod(ti)
    top = np.prod(to)
    tbp = np.prod(tb)

    # Check lengths and values.
    if ti.size != 3:
        raise ValueError("LoopBlocking: wrong length for ti.")
    if to.size != 3:
        raise ValueError("LoopBlocking: wrong length for to.")
    if tb.size != 3:
        raise ValueError("LoopBlocking: wrong length for tb.")

    class BL(object):  # pylint: disable=too-few-public-methods
        """
        Blocking-level enum. Only used locally.
        """

        GBUF = 0
        REGF = 1
        NUM = 2

    try:
        if tip < nested_loop_desc.loopcnt_ifm():
            raise ValueError("LoopBlocking: invalid blocking for ifm: {}".format(ti))
        if top < nested_loop_desc.loopcnt_ofm():
            raise ValueError("LoopBlocking: invalid blocking for ofm: {}".format(to))
        if tbp < nested_loop_desc.loopcnt_bat():
            raise ValueError("LoopBlocking: invalid blocking for bat: {}".format(tb))
    except Exception as e:
        return (float("inf"), None)

    ## Buffer data sizes in unit counts.

    cnt_units = [None for _ in range(BL.NUM)]
    for bl in range(BL.NUM):
        cnt_units[bl] = [0] * de.NUM
        cnt_units[bl][de.FIL] = np.prod(ti[bl + 1 :]) * np.prod(to[bl + 1 :])
        cnt_units[bl][de.IFM] = np.prod(ti[bl + 1 :]) * np.prod(tb[bl + 1 :])
        cnt_units[bl][de.OFM] = np.prod(to[bl + 1 :]) * np.prod(tb[bl + 1 :])

    ## Num ops, time, etc.

    lcnt_total = tip * top * tbp

    ops_total = nested_loop_desc.unit_num_ops() * lcnt_total

    time_total = nested_loop_desc.unit_time() * lcnt_total

    ## Basic size and reuse.

    assert BL.GBUF == 0
    assert BL.REGF == 1
    unit_size = [
        [x for x in nested_loop_desc.usize_gbuf()],
        [x for x in nested_loop_desc.usize_regf()],
    ]
    reuse = [None for _ in range(BL.NUM)]
    for bl in range(BL.NUM):
        reuse[bl] = [0] * de.NUM
        reuse[bl][de.FIL] = np.prod(tb[bl + 1 :])
        reuse[bl][de.IFM] = np.prod(to[bl + 1 :])
        reuse[bl][de.OFM] = np.prod(ti[bl + 1 :])

    ## Adjusted size and reuse based on loop orders, bypass, etc..

    size = [None] * BL.NUM

    def adjust_reuse(reuse_, bl_cur, order_cur, bls_outer, orders_outer):
        """
        Adjust the data reuse based on special loop structures.

        reuse_ is the reuse numbers for a specific level, e.g., reuse[BL.REGF].

        This function is recursive as we need to look at the outer levels.
        """
        if ti[bl_cur] != 1 and to[bl_cur] != 1:
            if order_cur.index(de.IFM) < order_cur.index(de.OFM):
                # Loop ifm inside loop ofm.
                # ofm also reused across current-level ifms.
                reuse_[de.OFM] *= ti[bl_cur]
            else:
                # Loop ifm outside loop ofm.
                # ifm also reused across current-level ofms.
                reuse_[de.IFM] *= to[bl_cur]
        elif ti[bl_cur] == 1 and to[bl_cur] != 1:
            # Current level does not change ifm, so ifm reuses ofms.
            reuse_[de.IFM] *= to[bl_cur]
        elif ti[bl_cur] != 1 and to[bl_cur] == 1:
            # Current level does not change ofm, so ofm reuses ifms.
            reuse_[de.OFM] *= ti[bl_cur]
        else:
            assert ti[bl_cur] == 1 and to[bl_cur] == 1
            # Current level loop counts are both 1 for ifms and ofms.
            # Effectively this level does not change the buffered data in the
            # inner level.
            # See the outer level.
            assert len(bls_outer) == len(orders_outer)
            if len(bls_outer) > 0:
                adjust_reuse(
                    reuse_,
                    bls_outer[0],
                    orders_outer[0],
                    bls_outer[1:],
                    orders_outer[1:],
                )

    # regf.
    adjust_reuse(reuse[BL.REGF], BL.REGF, orders[me.REGF], [BL.GBUF], [orders[me.GBUF]])

    size[BL.REGF] = [
        np.prod(tuple_) for tuple_ in zip(unit_size[BL.REGF], cnt_units[BL.REGF])
    ]
    if sum(size[BL.REGF]) > resource.size_regf:
        return (float("inf"), None)

    # gbuf.
    adjust_reuse(reuse[BL.GBUF], BL.GBUF, orders[me.GBUF], [], [])

    stored_in_gbuf = [1] * de.NUM
    # Only store in gbuf if having reuse.
    for deum in range(de.NUM):
        stored_in_gbuf[deum] = (
            1
            if not options.allow_gbuf_bypass[deum]
            or reuse[BL.GBUF][deum] > reuse[BL.REGF][deum]
            else 0
        )

    size[BL.GBUF] = [
        np.prod(tuple_)
        for tuple_ in zip(unit_size[BL.GBUF], cnt_units[BL.GBUF], stored_in_gbuf)
    ]
    if sum(size[BL.GBUF]) > resource.size_gbuf:
        return (float("inf"), None)

    ## Access.

    access = [0] * me.NUM

    access[me.REGF] = [v * lcnt_total for v in nested_loop_desc.unit_access(me.REGF)]

    access[me.ITCN] = [
        v * lcnt_total // r
        for v, r in zip(nested_loop_desc.unit_access(me.ITCN), reuse[BL.REGF])
    ]

    access[me.GBUF] = [
        v * lcnt_total // r * s
        for v, r, s in zip(
            nested_loop_desc.unit_access(me.GBUF), reuse[BL.REGF], stored_in_gbuf
        )
    ]

    access[me.DRAM] = [
        v * lcnt_total // r
        for v, r in zip(nested_loop_desc.unit_access(me.DRAM), reuse[BL.GBUF])
    ]

    ## Cost.

    access_total = [sum(a) for a in access]
    cost_loop = (
        np.dot(cost.memhier(), access_total)
        + ops_total * cost.macop()
        + time_total * cost.unit_static()
    )

    dict_loop = {
        "ops": ops_total,
        "time": time_total,
        "access": access,
        "size": size,
        "unit_size": unit_size,
        "ti": tuple(ti),
        "to": tuple(to),
        "tb": tuple(tb),
        "orders": orders,
    }

    return (cost_loop, dict_loop)

    # comb_lpl_part2d
    layer_data_size = [0] * de.NUM
    layer_data_size[de.FIL] = layer_part.total_filter_size()
    layer_data_size[de.IFM] = layer_part.total_ifmap_size() * batch_size
    layer_data_size[de.OFM] = layer_part.total_ofmap_size() * batch_size

    def sweep():  # pylint: disable=missing-docstring

        for cost_loop, dict_loop in _get_loopblocking_genfunc(options, layer_lb)(
            resource, cost, nested_loop_desc, options, layer_lb
        ):

            if math.isinf(cost_loop):
                continue

            # Calculate total number of hops, which determined by both
            # partition scheme (number of hops per access) and loop blocking
            # (runs of access).
            access_mem = dict_loop["access"][me.DRAM]
            assert len(access_mem) == de.NUM
            access_runs = [a / float(s) for a, s in zip(access_mem, layer_data_size)]
            total_nhops = [nh * r for nh, r in zip(unit_nhops, access_runs)]

            # Partition cost.
            cost_part = cost.nochop() * sum(total_nhops)
            dict_part = {
                "unit_nhops": unit_nhops,
                "total_nhops": total_nhops,
                "part_lprev": part_lprev.as_pod_type(),
                "part_lcurr": part_lcurr.as_pod_type(),
            }

            # Combine.
            dict_loop.update({"cost": cost_loop})
            dict_part.update({"cost": cost_part})
            yield (
                cost_loop * resource.dim_nodes.size() + cost_part,
                dict_loop,
                dict_part,
            )

    return heapq.nsmallest(options.ntops, sweep(), key=lambda x: x[0])


def factorize(value, num, limits=None):
    '''
    Factorize given `value` into `num` numbers. Return as a copy of num-length
    np.array.

    Iterate over factor combinations of which the product is `value`.

    `limits` is a (num-1)-length tuple, specifying the upper limits for the
    first num-1 factors.
    '''
    if limits is None:
        limits = [float('inf')] * (num - 1)
    assert len(limits) >= num - 1
    limits = limits[:num-1] + [float('inf')]

    factors = np.ones(num, dtype=int)
    while True:
        # Calculate the last factor.
        factors[-1] = idivc(value, np.prod(factors[:-1]))
        if np.prod(factors) == value \
                and np.all(np.less(factors, limits)):
            yield tuple(np.copy(factors))

        # Update the first n - 1 factor combination, backwards.
        lvl = num - 1
        while lvl >= 0:
            factors[lvl] += 1
            if np.prod(factors[:lvl+1]) <= value:
                break
            else:
                factors[lvl] = 1
                lvl -= 1
        if lvl < 0:
            return

def closest_factor(value, factor):
    '''
    Return the maximum factor of `value` that is no larger than `factor`, and
    the minimum factor of `value` that is no less than `factor`, as a tuple.
    '''
    res = tuple()

    # Maximum no-larger factor.
    f = int(factor)
    while f > 1:
        if value % f == 0 and f <= factor:
            break
        f -= 1
    res += (max(1, f), )

    # Minimum no-smaller factor.
    f = int(factor)
    while f < value:
        if f != 0 and value % f == 0 and f >= factor:
            break
        f += 1
    res += (min(value, f), )

    return res

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
