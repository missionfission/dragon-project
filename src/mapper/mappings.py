from mapper import Mapper
from synthesis.dataflow_solver import dataflow_solver_wrapper

"""

Provides Mapping Interface for 5 types of mapping :

1. ASAP Mapping

2. Energy Efficienct Mapping

3. Energy Efficienct Mapping with Power Gating

4. NN-Dataflow Mapping

5. Illusion Mapping for Multiple Chips
"""


def run_asap(self, graph):

    """
    Runs the Graph on the Hardware ASAP Mapped

    We following a Dynamic State Variable Execution Mapping :
    1. Memory Size and Maximum Allowed Bandwidth are taken from Hardware Config
    2. Current Memory Size and Memory Utilization are calculated by Mapping the Nodes Serially

    Memory Management Scenarios :

    1. Check both size, utilization and bandwidths at every node
    2. What about memory size that can also get exhausted 
    3. If memory size is exhausted, then to go to a previous level and write there 
    4. If any level utilization is exhausted then only the immediate memory required will be kept.
    5. If the memory is empty in size, but there is no bandwidth, it is useless : Cannot do prefetching
    6. If Prefetching : Read access of the next node will decrease
    7. Bandwidth is available but size is not : Can do prefetching, but now the memory fetches have to check,
    whether to do fetches of the same node or a different node
    8. Say bandwidth at level0 is sufficient, at level1 is insufficient, then at level1 we have a bottlenecks
    slower so it will take its own time

    Compute Management Scenarios :

    1. Pipelined vs Parallel Mapper 
    2. When do vector operations happen 
    3. Scale up vs Scale out for Systolic Arrays

    """

    config = self.config
    # TODO in_edge_mem can change by pooling/batch_norm
    # TODO compute is very much higher than bandwidth time and mem size idle time
    read_bw_req = []
    write_bw_req = []
    read_bw_actual = []
    write_bw_actual = []
    cycles = []
    free_cycles = []
    transferable_checkpointed_edge = []
    all_checkpointed_edge = []
    self.mem_util_log = []
    self.mem_util_full = []
    # Mem Fetch time of the last Nodes
    #     print(self.mem_free[0], self.mem_util[0], self.mem_size[0])

    mem_free = True
    for n, node in enumerate(graph.nodes):
        node.mem_fetch = node.weights
    for n, node in enumerate(graph.nodes):

        # These are last level read/write accesses
        compute_expense, weights = node.get_stats()
        read_access = node.mem_fetch
        write_access = 0
        self.mem_read_access[1] += weights

        assert self.mem_util[0] <= self.mem_size[0]
        self.mem_util[0] += node.in_edge_mem
        node.mem_util = node.out_edge_mem + node.mem_fetch
        # Total Free memory
        for i in range(self.mle - 1):
            self.mem_free[i] = self.mem_size[i] - self.mem_util[i]
        time_compute = compute_expense / config["mm_compute_per_cycle"]
        read_bw_ll = read_access / (time_compute)
        write_bw_ll = write_access / (time_compute)
        step_cycles = time_compute
        read_bw_req.append(read_bw_ll)
        write_bw_req.append(write_bw_ll)
        free_cycles.append(step_cycles)
        n_swaps = 1
        total_mem = 0
        if self.mem_free[0] < node.mem_util:
            mem_free = False
            self.mem_util[0] -= node.in_edge_mem
            self.mem_util[0] -= node.weights - node.mem_fetch
            self.mem_free[0] = self.mem_size[0] - self.mem_util[0]
            total_mem = node.in_edge_mem + node.out_edge_mem + node.weights
            if self.mem_free[0] <= 0:
                print(self.mem_free[0])
            assert self.mem_free[0] > 0, self.mem_util[0]
            n_swaps = total_mem // self.mem_free[0] + 1
            swap_time = max(config["mm_compute"]["size"] * 4, time_compute // n_swaps)
            self.mem_size_idle_time += (
                swap_time * n_swaps
                + ((node.out_edge_mem // n_swaps - 1) * n_swaps)
                // self.mem_read_bw[self.mle - 1]
            )
            self.bandwidth_idle_time += (
                (node.out_edge_mem // n_swaps - 1) * n_swaps
            ) // self.mem_read_bw[self.mle - 1]
            step_cycles += (
                swap_time * n_swaps
                + 2
                * ((node.out_edge_mem // n_swaps - 1) * n_swaps)
                // self.mem_read_bw[self.mle - 1]
            )
            self.mem_read_access[0] += node.mem_util + node.in_edge_mem
            self.mem_write_access[0] += node.mem_util + node.in_edge_mem
            self.mem_write_access[1] += node.out_edge_mem
        else:
            self.mem_util[0] += node.mem_util
            self.mem_free[0] -= node.mem_util
        #         print("2.5",self.mem_free[0], self.mem_util[0], self.mem_size[0])
        self.mem_util_log.append(self.mem_util[0])
        self.mem_read_access[0] += node.weights + node.out_edge_mem
        self.mem_write_access[0] += node.weights + node.out_edge_mem
        # assert self.mem_free[0] <= self.mem_size[0]
        # Last level memory fetch takes more time, so that may be a bottleneck
        bandwidth_available = read_bw_ll < self.mem_read_bw[self.mle - 1]

        # If Bandwidth is not available : Cannot Prefetch
        if (bandwidth_available) == False:
            step_cycles += (
                read_bw_ll / self.mem_read_bw[self.mle - 1] - 1
            ) * time_compute
            self.bandwidth_idle_time += (
                read_bw_ll / self.mem_read_bw[self.mle - 1] - 1
            ) * time_compute

        # If memory is not free for the next node and Bandwidth is available : Move nodes back and forth
        # if(total_mem_free[0] == 0 and (bandwidth_available)):
        # for(nodes in checkpointed_nodes):
        # checkpointed but not immediate node

        # Check if memory is free and Bandwidth available : From the Data Dependence Graph, Prefetch new node

        # pdb.set_trace()
        if self.mem_free[0] > 0 and (bandwidth_available):
            if n < len(graph.nodes) - 1:
                if self.mem_free[0] > node.next.mem_fetch:
                    read_access += node.next.mem_fetch
                    if read_access / step_cycles < self.mem_read_bw[self.mle - 1]:
                        self.mem_util[0] += node.next.mem_fetch
                        self.mem_free[0] -= node.next.mem_fetch
                        node.next.mem_fetch = 0
                    else:
                        read_access = self.mem_read_bw[self.mle - 1] * step_cycles
                        self.mem_util[0] += read_access - read_bw_ll * step_cycles
                        self.mem_free[0] -= read_access - read_bw_ll * step_cycles
                        node.next.mem_fetch -= (
                            read_access - read_bw_ll * step_cycles
                        )  # Next node mem fetch gets updated

                else:
                    read_access += self.mem_free[0]
                    if read_access / step_cycles < self.mem_read_bw[self.mle - 1]:
                        node.next.mem_fetch = node.next.mem_fetch - self.mem_free[0]
                        # Next node mem fetch gets updated

                        self.mem_util[0] = self.mem_size[0]
                        self.mem_free[0] = 0
                    else:
                        read_access = self.mem_read_bw[self.mle - 1] * step_cycles
                        self.mem_util[0] += read_access - read_bw_ll * step_cycles
                        self.mem_free[0] -= read_access - read_bw_ll * step_cycles
                        node.next.mem_fetch -= read_access - read_bw_ll * step_cycles
                        # Next node mem fetch gets updated

        #         print("3",self.mem_free[0], self.mem_util[0], self.mem_size[0])
        self.mem_util_full.append(self.mem_util[0])

        # TODO Consider Write bandwidth for a block read memory or Write Bandwidth for endurance purposes
        #         print("4",self.mem_free[0], self.mem_util[0], self.mem_size[0])

        if mem_free:
            self.mem_util[0] -= node.out_edge_mem + node.weights + node.in_edge_mem
        #         print("5",self.mem_free[0], self.mem_util[0], self.mem_size[0])

        self.logger.debug(
            "Node operator %r, Compute Expense %d,   Time Compute %d, Step Cycles %d, Read Accesses %d, Write Accesses %d , No of Swaps %d, Total_mem %d",
            node.operator,
            compute_expense,
            time_compute,
            step_cycles,
            read_access,
            write_access,
            n_swaps,
            total_mem,
        )
        self.total_cycles += step_cycles
        cycles.append(step_cycles)
        read_bw_actual.append(read_access / step_cycles)
        write_bw_actual.append(write_access / step_cycles)
    #         print("actual",read_access / step_cycles, write_access / step_cycles, step_cycles)
    #     print("The total cycles are ", self.total_cycles)
    self.mem_write_access[1] += node.out_edge_mem
    return (
        read_bw_req,
        write_bw_req,
        read_bw_actual,
        write_bw_actual,
        cycles,
        free_cycles,
    )


def run_reuse_leakage(self, graph):
    """
    Run an ASAP Mapping and choose the greedy choice between Reuse and Leakage_Power
    
    Implementation :
        1. Quantify the Scenarios of Reuse
        2. Time taken by the Loop Blocking for Reuse

    """

    config = self.config
    # TODO in_edge_mem can change by pooling/batch_norm
    # TODO compute is very much higher than bandwidth time and mem size idle time
    read_bw_req = []
    write_bw_req = []
    read_bw_actual = []
    write_bw_actual = []
    cycles = []
    free_cycles = []
    transferable_checkpointed_edge = []
    all_checkpointed_edge = []
    self.mem_util_log = []
    self.mem_util_full = []
    # Mem Fetch time of the last Nodes
    #     print(self.mem_free[0], self.mem_util[0], self.mem_size[0])

    mem_free = True
    for n, node in enumerate(graph.nodes):
        node.mem_fetch = node.weights
        # These are last level read/write accesses
        compute_expense, weights = node.get_stats()
        """
        mem_reuse, comp_reuse = get_reuse(node)
        time_reuse =         # reusing memory : then time_reuse = time_taken - time_without_reuse
        
        """
        read_access = node.mem_fetch
        write_access = 0
        self.mem_read_access[1] += weights

        assert self.mem_util[0] <= self.mem_size[0]
        self.mem_util[0] += node.in_edge_mem
        node.mem_util = node.out_edge_mem + node.mem_fetch
        # Total Free memory
        for i in range(self.mle - 1):
            self.mem_free[i] = self.mem_size[i] - self.mem_util[i]
        time_compute = compute_expense / config["mm_compute_per_cycle"]
        read_bw_ll = read_access / (time_compute)
        write_bw_ll = write_access / (time_compute)
        step_cycles = time_compute
        read_bw_req.append(read_bw_ll)
        write_bw_req.append(write_bw_ll)
        free_cycles.append(step_cycles)
        n_swaps = 1
        total_mem = 0
        if self.mem_free[0] < node.mem_util:
            mem_free = False
            self.mem_util[0] -= node.in_edge_mem
            self.mem_util[0] -= node.weights - node.mem_fetch
            self.mem_free[0] = self.mem_size[0] - self.mem_util[0]
            total_mem = node.in_edge_mem + node.out_edge_mem + node.weights
            if self.mem_free[0] <= 0:
                print(self.mem_free[0])
            assert self.mem_free[0] > 0, self.mem_util[0]
            n_swaps = total_mem // self.mem_free[0] + 1
            swap_time = max(config["mm_compute"]["size"] * 4, time_compute // n_swaps)
            self.mem_size_idle_time += (
                swap_time * n_swaps
                + ((node.out_edge_mem // n_swaps - 1) * n_swaps)
                // self.mem_read_bw[self.mle - 1]
            )
            self.bandwidth_idle_time += (
                (node.out_edge_mem // n_swaps - 1) * n_swaps
            ) // self.mem_read_bw[self.mle - 1]
            step_cycles += (
                swap_time * n_swaps
                + 2
                * ((node.out_edge_mem // n_swaps - 1) * n_swaps)
                // self.mem_read_bw[self.mle - 1]
            )
            self.mem_read_access[0] += node.mem_util + node.in_edge_mem
            self.mem_write_access[0] += node.mem_util + node.in_edge_mem
            self.mem_write_access[1] += node.out_edge_mem
        else:
            self.mem_util[0] += node.mem_util
            self.mem_free[0] -= node.mem_util
        #         print("2.5",self.mem_free[0], self.mem_util[0], self.mem_size[0])
        self.mem_util_log.append(self.mem_util[0])
        self.mem_read_access[0] += node.weights + node.out_edge_mem
        self.mem_write_access[0] += node.weights + node.out_edge_mem
        assert self.mem_free[0] <= self.mem_size[0]
        # Last level memory fetch takes more time, so that may be a bottleneck
        bandwidth_available = read_bw_ll < self.mem_read_bw[self.mle - 1]

        # If Bandwidth is not available : Cannot Prefetch
        if (bandwidth_available) == False:
            step_cycles += (
                read_bw_ll / self.mem_read_bw[self.mle - 1] - 1
            ) * time_compute
            self.bandwidth_idle_time += (
                read_bw_ll / self.mem_read_bw[self.mle - 1] - 1
            ) * time_compute

        # If memory is not free for the next node and Bandwidth is available : Move nodes back and forth
        # if(total_mem_free[0] == 0 and (bandwidth_available)):
        # for(nodes in checkpointed_nodes):
        # checkpointed but not immediate node

        # Check if memory is free and Bandwidth available : From the Data Dependence Graph, Prefetch new node

        # pdb.set_trace()
        if self.mem_free[0] > 0 and (bandwidth_available):
            if n < len(graph.nodes) - 1:
                if self.mem_free[0] > node.next.mem_fetch:
                    read_access += node.next.mem_fetch
                    if read_access / step_cycles < self.mem_read_bw[self.mle - 1]:
                        self.mem_util[0] += node.next.mem_fetch
                        self.mem_free[0] -= node.next.mem_fetch
                        node.next.mem_fetch = 0
                    else:
                        read_access = self.mem_read_bw[self.mle - 1] * step_cycles
                        self.mem_util[0] += read_access - read_bw_ll * step_cycles
                        self.mem_free[0] -= read_access - read_bw_ll * step_cycles
                        node.next.mem_fetch -= (
                            read_access - read_bw_ll * step_cycles
                        )  # Next node mem fetch gets updated

                else:
                    read_access += self.mem_free[0]
                    if read_access / step_cycles < self.mem_read_bw[self.mle - 1]:
                        node.next.mem_fetch = node.next.mem_fetch - self.mem_free[0]
                        # Next node mem fetch gets updated

                        self.mem_util[0] = self.mem_size[0]
                        self.mem_free[0] = 0
                    else:
                        read_access = self.mem_read_bw[self.mle - 1] * step_cycles
                        self.mem_util[0] += read_access - read_bw_ll * step_cycles
                        self.mem_free[0] -= read_access - read_bw_ll * step_cycles
                        node.next.mem_fetch -= read_access - read_bw_ll * step_cycles
                        # Next node mem fetch gets updated

        #         print("3",self.mem_free[0], self.mem_util[0], self.mem_size[0])
        self.mem_util_full.append(self.mem_util[0])

        # TODO Consider Write bandwidth for a block read memory or Write Bandwidth for endurance purposes
        #         print("4",self.mem_free[0], self.mem_util[0], self.mem_size[0])

        if mem_free:
            self.mem_util[0] -= node.out_edge_mem + node.weights + node.in_edge_mem
        #         print("5",self.mem_free[0], self.mem_util[0], self.mem_size[0])

        self.logger.debug(
            "Node operator %r, Compute Expense %d,   Time Compute %d, Step Cycles %d, Read Accesses %d, Write Accesses %d , No of Swaps %d, Total_mem %d",
            node.operator,
            compute_expense,
            time_compute,
            step_cycles,
            read_access,
            write_access,
            n_swaps,
            total_mem,
        )
        self.total_cycles += step_cycles
        cycles.append(step_cycles)
        read_bw_actual.append(read_access / step_cycles)
        write_bw_actual.append(write_access / step_cycles)
    #         print("actual",read_access / step_cycles, write_access / step_cycles, step_cycles)
    #     print("The total cycles are ", self.total_cycles)
    self.mem_write_access[1] += node.out_edge_mem
    return (
        read_bw_req,
        write_bw_req,
        read_bw_actual,
        write_bw_actual,
        cycles,
        free_cycles,
    )


def run_reuse_full(self, graph):
    """
    Run an ASAP Mapping and allow maximal reuse and fine-grained power gating of the components 
    """
    config = self.config
    # TODO in_edge_mem can change by pooling/batch_norm
    # TODO compute is very much higher than bandwidth time and mem size idle time
    read_bw_req = []
    write_bw_req = []
    read_bw_actual = []
    write_bw_actual = []
    cycles = []
    free_cycles = []
    transferable_checkpointed_edge = []
    all_checkpointed_edge = []
    self.mem_util_log = []
    self.mem_util_full = []
    # Mem Fetch time of the last Nodes
    #     print(self.mem_free[0], self.mem_util[0], self.mem_size[0])

    mem_free = True
    for n, node in enumerate(graph.nodes):
        node.mem_fetch = node.weights
    for n, node in enumerate(graph.nodes):

        # These are last level read/write accesses
        compute_expense, weights = node.get_stats()
        """
        mem_reuse, comp_reuse = get_reuse(node)
        time_reuse = 
        # reusing memory : then time_reuse = time_taken - time_without_reuse
        total_leakage_power = time_reuse *() # sum of leakage power of all
        is_reuse = True
        if(total_leakage_power > mem_reuse*mem_energy):
            is_reuse = False
        """
        read_access = node.mem_fetch
        write_access = 0
        self.mem_read_access[1] += weights

        assert self.mem_util[0] <= self.mem_size[0]
        self.mem_util[0] += node.in_edge_mem
        node.mem_util = node.out_edge_mem + node.mem_fetch
        # Total Free memory
        for i in range(self.mle - 1):
            self.mem_free[i] = self.mem_size[i] - self.mem_util[i]
        time_compute = compute_expense / config["mm_compute_per_cycle"]
        read_bw_ll = read_access / (time_compute)
        write_bw_ll = write_access / (time_compute)
        step_cycles = time_compute
        read_bw_req.append(read_bw_ll)
        write_bw_req.append(write_bw_ll)
        free_cycles.append(step_cycles)
        n_swaps = 1
        total_mem = 0
        if self.mem_free[0] < node.mem_util:
            mem_free = False
            self.mem_util[0] -= node.in_edge_mem
            self.mem_util[0] -= node.weights - node.mem_fetch
            self.mem_free[0] = self.mem_size[0] - self.mem_util[0]
            total_mem = node.in_edge_mem + node.out_edge_mem + node.weights
            if self.mem_free[0] <= 0:
                print(self.mem_free[0])
            assert self.mem_free[0] > 0, self.mem_util[0]
            n_swaps = total_mem // self.mem_free[0] + 1
            swap_time = max(config["mm_compute"]["size"] * 4, time_compute // n_swaps)
            self.mem_size_idle_time += (
                swap_time * n_swaps
                + ((node.out_edge_mem // n_swaps - 1) * n_swaps)
                // self.mem_read_bw[self.mle - 1]
            )
            self.bandwidth_idle_time += (
                (node.out_edge_mem // n_swaps - 1) * n_swaps
            ) // self.mem_read_bw[self.mle - 1]
            step_cycles += (
                swap_time * n_swaps
                + 2
                * ((node.out_edge_mem // n_swaps - 1) * n_swaps)
                // self.mem_read_bw[self.mle - 1]
            )
            self.mem_read_access[0] += node.mem_util + node.in_edge_mem
            self.mem_write_access[0] += node.mem_util + node.in_edge_mem
            self.mem_write_access[1] += node.out_edge_mem
        else:
            self.mem_util[0] += node.mem_util
            self.mem_free[0] -= node.mem_util
        #         print("2.5",self.mem_free[0], self.mem_util[0], self.mem_size[0])
        self.mem_util_log.append(self.mem_util[0])
        self.mem_read_access[0] += node.weights + node.out_edge_mem
        self.mem_write_access[0] += node.weights + node.out_edge_mem
        assert self.mem_free[0] <= self.mem_size[0]
        # Last level memory fetch takes more time, so that may be a bottleneck
        bandwidth_available = read_bw_ll < self.mem_read_bw[self.mle - 1]

        # If Bandwidth is not available : Cannot Prefetch
        if (bandwidth_available) == False:
            step_cycles += (
                read_bw_ll / self.mem_read_bw[self.mle - 1] - 1
            ) * time_compute
            self.bandwidth_idle_time += (
                read_bw_ll / self.mem_read_bw[self.mle - 1] - 1
            ) * time_compute

        # If memory is not free for the next node and Bandwidth is available : Move nodes back and forth
        # if(total_mem_free[0] == 0 and (bandwidth_available)):
        # for(nodes in checkpointed_nodes):
        # checkpointed but not immediate node

        # Check if memory is free and Bandwidth available : From the Data Dependence Graph, Prefetch new node

        # pdb.set_trace()
        if self.mem_free[0] > 0 and (bandwidth_available):
            if n < len(graph.nodes) - 1:
                if self.mem_free[0] > node.next.mem_fetch:
                    read_access += node.next.mem_fetch
                    if read_access / step_cycles < self.mem_read_bw[self.mle - 1]:
                        self.mem_util[0] += node.next.mem_fetch
                        self.mem_free[0] -= node.next.mem_fetch
                        node.next.mem_fetch = 0
                    else:
                        read_access = self.mem_read_bw[self.mle - 1] * step_cycles
                        self.mem_util[0] += read_access - read_bw_ll * step_cycles
                        self.mem_free[0] -= read_access - read_bw_ll * step_cycles
                        node.next.mem_fetch -= (
                            read_access - read_bw_ll * step_cycles
                        )  # Next node mem fetch gets updated

                else:
                    read_access += self.mem_free[0]
                    if read_access / step_cycles < self.mem_read_bw[self.mle - 1]:
                        node.next.mem_fetch = node.next.mem_fetch - self.mem_free[0]
                        # Next node mem fetch gets updated

                        self.mem_util[0] = self.mem_size[0]
                        self.mem_free[0] = 0
                    else:
                        read_access = self.mem_read_bw[self.mle - 1] * step_cycles
                        self.mem_util[0] += read_access - read_bw_ll * step_cycles
                        self.mem_free[0] -= read_access - read_bw_ll * step_cycles
                        node.next.mem_fetch -= read_access - read_bw_ll * step_cycles
                        # Next node mem fetch gets updated

        #         print("3",self.mem_free[0], self.mem_util[0], self.mem_size[0])
        self.mem_util_full.append(self.mem_util[0])

        # TODO Consider Write bandwidth for a block read memory or Write Bandwidth for endurance purposes
        #         print("4",self.mem_free[0], self.mem_util[0], self.mem_size[0])

        if mem_free:
            self.mem_util[0] -= node.out_edge_mem + node.weights + node.in_edge_mem
        #         print("5",self.mem_free[0], self.mem_util[0], self.mem_size[0])

        self.logger.debug(
            "Node operator %r, Compute Expense %d,   Time Compute %d, Step Cycles %d, Read Accesses %d, Write Accesses %d , No of Swaps %d, Total_mem %d",
            node.operator,
            compute_expense,
            time_compute,
            step_cycles,
            read_access,
            write_access,
            n_swaps,
            total_mem,
        )
        self.total_cycles += step_cycles
        cycles.append(step_cycles)
        read_bw_actual.append(read_access / step_cycles)
        write_bw_actual.append(write_access / step_cycles)
    #         print("actual",read_access / step_cycles, write_access / step_cycles, step_cycles)
    #     print("The total cycles are ", self.total_cycles)
    self.mem_write_access[1] += node.out_edge_mem
    return (
        read_bw_req,
        write_bw_req,
        read_bw_actual,
        write_bw_actual,
        cycles,
        free_cycles,
    )


def run_nn_dataflow(self, graph):
    config = self.config
    # TODO in_edge_mem can change by pooling/batch_norm
    # TODO compute is very much higher than bandwidth time and mem size idle time
    read_bw_req = []
    write_bw_req = []
    read_bw_actual = []
    write_bw_actual = []
    cycles = []
    free_cycles = []
    transferable_checkpointed_edge = []
    all_checkpointed_edge = []
    self.mem_util_log = []
    self.mem_util_full = []
    # Mem Fetch time of the last Nodes
    #     print(self.mem_free[0], self.mem_util[0], self.mem_size[0])
    step_cycles = 0
    mem_free = True
    for n, node in enumerate(graph.nodes):
        node.mem_fetch = node.weights
    
    for n, node in enumerate(graph.nodes):

        # These are last level read/write accesses
        compute_expense, weights = node.get_stats()
        """
        mem_reuse, comp_reuse = get_reuse(node)
        time_reuse = time_taken - time_without_reuse
        total_leakage_power = time_reuse *() # sum of leakage power of all
        is_reuse = True
        if(total_leakage_power > mem_reuse*mem_energy):
            is_reuse = False
        """
        read_access = node.mem_fetch
        write_access = 0
        self.mem_read_access[1] += weights

        assert self.mem_util[0] <= self.mem_size[0]
        self.mem_util[0] += node.in_edge_mem
        node.mem_util = node.out_edge_mem + node.mem_fetch
        # Total Free memory
        for i in range(self.mle - 1):
            self.mem_free[i] = self.mem_size[i] - self.mem_util[i]
        time_compute = dataflow_solver_wrapper(self, node)
        step_cycles += time_compute
        read_bw_ll = read_access / (time_compute)
        write_bw_ll = write_access / (time_compute)
        step_cycles = time_compute
        read_bw_req.append(read_bw_ll)
        write_bw_req.append(write_bw_ll)
        free_cycles.append(step_cycles)
        n_swaps = 1
        total_mem = 0
        self.mem_read_access[0] += node.mem_util + node.in_edge_mem
        self.mem_write_access[0] += node.mem_util + node.in_edge_mem
        self.mem_write_access[1] += node.out_edge_mem
        self.mem_util[0] += node.mem_util
        self.mem_free[0] -= node.mem_util
        self.mem_util_log.append(self.mem_util[0])
        self.mem_read_access[0] += node.weights + node.out_edge_mem
        self.mem_write_access[0] += node.weights + node.out_edge_mem
        assert self.mem_free[0] <= self.mem_size[0]
        # Last level memory fetch takes more time, so that may be a bottleneck
        step_cycles += read_access / self.mem_read_bw[self.mle - 1] + 1
        self.bandwidth_idle_time += read_access / self.mem_read_bw[self.mle - 1] + 1

        self.mem_util_full.append(self.mem_util[0])

        # TODO Consider Write bandwidth for a block read memory or Write Bandwidth for endurance purposes
        #         print("4",self.mem_free[0], self.mem_util[0], self.mem_size[0])

        if mem_free:
            self.mem_util[0] -= node.out_edge_mem + node.weights + node.in_edge_mem
        #         print("5",self.mem_free[0], self.mem_util[0], self.mem_size[0])

        self.logger.debug(
            "Node operator %r, Compute Expense %d,   Time Compute %d, Step Cycles %d, Read Accesses %d, Write Accesses %d , No of Swaps %d, Total_mem %d",
            node.operator,
            compute_expense,
            time_compute,
            step_cycles,
            read_access,
            write_access,
            n_swaps,
            total_mem,
        )
        self.total_cycles += step_cycles
        cycles.append(step_cycles)
        read_bw_actual.append(read_access / step_cycles)
        write_bw_actual.append(write_access / step_cycles)
        #         print("actual",read_access / step_cycles, write_access / step_cycles, step_cycles)
        #     print("The total cycles are ", self.total_cycles)
        self.mem_write_access[1] += node.out_edge_mem
    # total_fetch = 0
    # for i, node in enumerate(graph.nodes):
    #     total_fetch += (node.in_edge_mem) // 2
    # step_cycles += total_fetch / self.mem_read_bw[self.mle - 1] + 1
    # self.bandwidth_idle_time += total_fetch / self.mem_read_bw[self.mle - 1] + 1
    return (
        read_bw_req,
        write_bw_req,
        read_bw_actual,
        write_bw_actual,
        cycles,
        free_cycles,
    )


def illusion_mapping(graph, num_of_chips, depth, capacity, deeper=False, wider=False):
    """ 
    
    Illusion Mapping for Deeper and Wider Networks from Radway et. al. Nat Ele.'20

    Args:
        graph (): 
        num_of_chips (): 
        depth (): 
        capacity (): 
        deeper (bool, optional): . Defaults to False.
        wider (bool, optional): . Defaults to False.
    """
    mem_size_util = np.zeros((num_of_chips + 1))
    mem_free = np.zeros((num_of_chips + 1))
    message_passed = np.zeros((num_of_chips + 1))
    i = 0
    # mem_track_node
    if deeper:
        layer = 0
        for node in graph.nodes:
            for k in range(depth):
                if node.operator == "aten::_convolution":
                    layer += 1
                    if node.outputs[0].shape[1] == node.inputs[1].shape[0]:
                        oc, ic, *ks = node.inputs[1].shape
                    else:
                        ic, oc, *ks = node.inputs[1].shape
                    mem_size_util[i] += node.weights

                    if mem_size_util[i] < capacity:
                        continue
                    elif mem_size_util[i] > capacity:
                        mem_size_util[i] -= node.weights
                        # partition of node.weights
                        if np.prod(node.outputs[0].shape) > np.prod(
                            node.inputs[0].shape
                        ):
                            # weights : (oc,ic,x,y)
                            mem_free[i] = capacity - mem_size_util[i]
                            oc_partition = mem_free[i] // (np.prod(ks) * ic)
                            mem_size_util[i + 1] = (
                                node.weights
                            ) - oc_partition * np.prod(ks) * ic
                            # outputs : oc, ox, oy, inputs : ic,ix, iy

                            message_passed[i + 1] += np.prod(
                                node.outputs[0].shape[2:]
                            ) * oc_partition + np.prod(node.inputs[0].shape)

                        else:
                            mem_free[i] = capacity - mem_size_util[i]
                            ic_partition = mem_free[i] // (oc * np.prod(ks))
                            mem_size_util[
                                i + 1
                            ] = node.weights - ic_partition * oc * np.prod(ks)
                            message_passed[i + 1] += np.prod(
                                node.inputs[0].shape[2:]
                            ) * ic_partition + np.prod(node.outputs[0].shape)
                        i += 1
                        message_passed[i] += np.prod(node.outputs[0].shape)
                    # if node.operator == "aten::addmm":
                    #     mem_size_util[i] += node.weights

                    #     if mem_size_util[i] < capacity:
                    #         continue
                    #     mem_size_util[i] -= node.weights
                    #     n, m = node.inputs[1].shape
                    #     m, p = node.inputs[2].shape
                    #     if mem_size_util[i] + n * p > capacity:
                    #         message_passed[i] += n * p

                    #     if np.prod(node.inputs[0].shape) > np.prod(node.inputs[1].shape):
                    #         x = capacity // p
                    #         message_passed[i] += m * p + (n - x) * m
                    #         mem_size_util[i + 1] += (n - x) * p
                    #     else:
                    #         x = capacity // (n)
                    #         message_passed[i] += m * (p - x) + n * m
                    #         mem_size_util[i + 1] += (p - x) * n
                    #     i += 1
                    # if node.operator == "aten::bmm":
                    #     mem_size_util[i] += node.weights

                    #     if mem_size_util[i] < capacity:
                    #         continue
                    #     *b, n, p = node.outputs[0].shape
                    #     *_, n, m = node.inputs[0].shape
                    #     *_, m, p = node.inputs[1].shape
                    #     if np.prod(node.inputs[0].shape) > np.prod(node.inputs[1].shape):
                    #         x = capacity // (np.prod(b) * p)
                    #         message_passed[i] += (
                    #             np.prod(b) * m * p + np.prod(b) * (n - x) * m
                    #         )
                    #         mem_size_util[i + 1] += np.prod(b) * (n - x) * p

                    #     else:
                    #         x = capacity // (np.prod(b) * n)
                    #         message_passed[i] += (
                    #             np.prod(b) * m * (p - x) + np.prod(b) * n * m
                    #         )
                    #         mem_size_util[i + 1] += np.prod(b) * (p - x) * n
                    #     i += 1
                    # if node.operator == "aten::matmul":
                    #     mem_size_util[i] += node.weights

                    #     if mem_size_util[i] < capacity:
                    #         continue

                    #     if len(node.inputs) > 1:
                    #         if node.inputs[0].ndim == 2 and node.inputs[1].ndim == 2:
                    #             n, p = node.outputs[0].shape
                    #             n, m = node.inputs[0].shape
                    #             m, p = node.inputs[1].shape
                    #             if np.prod(node.inputs[0].shape) > np.prod(
                    #                 node.inputs[1].shape
                    #             ):
                    #                 x = capacity // (p)
                    #                 message_passed[i] += m * p + (n - x) * m
                    #                 mem_size_util[i + 1] += (n - x) * p
                    #             else:
                    #                 x = capacity // (n)
                    #                 message_passed[i] += m * (p - x) + n * m
                    #                 mem_size_util[i + 1] += (p - x) * n

                    #         elif node.inputs[0].ndim > 2 and node.inputs[1].ndim > 2:
                    #             *b, n, p = node.outputs[0].shape
                    #             *_, n, m = node.inputs[0].shape
                    #             *_, m, p = node.inputs[1].shape
                    #             if np.prod(node.inputs[0].shape) > np.prod(
                    #                 node.inputs[1].shape
                    #             ):
                    #                 x = capacity // (np.prod(b) * p)
                    #                 message_passed[i] += (
                    #                     np.prod(b) * m * p + np.prod(b) * (n - x) * m
                    #                 )
                    #                 mem_size_util[i + 1] += np.prod(b) * (n - x) * p
                    #             else:
                    #                 x = capacity // (np.prod(b) * n)
                    #                 message_passed[i] += (
                    #                     np.prod(b) * m * (p - x) + np.prod(b) * n * m
                    #                 )
                    #                 mem_size_util[i + 1] += np.prod(b) * (p - x) * n
                    # i += 1
        print(np.sum(message_passed))

    if wider:
        for node in graph.nodes:
            layer = 0
            if node.operator == "aten::_convolution":
                layer += 1
                if node.outputs[0].shape[1] == node.inputs[1].shape[0]:
                    oc, ic, *ks = node.inputs[1].shape
                else:
                    ic, oc, *ks = node.inputs[1].shape
                # depth first for residual blocks

                mem_size_util[i] += depth * node.weights
                if mem_size_util[i] < capacity:
                    continue
                elif mem_size_util[i] > capacity:
                    mem_size_util[i] -= depth * node.weights
                    # partition of node.weights
                    # print("Layer partition", layer)
                    if np.prod(node.outputs[0].shape) > np.prod(node.inputs[0].shape):
                        # weights : (oc,ic,x,y)
                        mem_free[i] = capacity - mem_size_util[i]
                        oc_partition = mem_free[i] // (depth * np.prod(ks) * ic)
                        mem_size_util[i + 1] = (
                            depth * (node.weights)
                            - depth * oc_partition * np.prod(ks) * ic
                        )
                        # outputs : oc, ox, oy, inputs : ic,ix, iy
                        # message_passed[i + 1] = np.prod(
                        #     node.outputs[0].shape[2:]
                        # ) * oc_partition + np.prod(node.inputs[0].shape)
                        if mem_size_util[i + 1] > capacity:
                            depth_ratio = (
                                depth
                                * ((node.weights) - oc_partition * np.prod(ks) * ic)
                            ) // capacity

                            depth_left = (
                                depth
                                * ((node.weights) - oc_partition * np.prod(ks) * ic)
                            ) % capacity
                            message_passed[i + 1] = depth_ratio * (
                                np.prod(node.outputs[0].shape)
                                + np.prod(node.inputs[0].shape)
                            )

                            i += depth_ratio
                            mem_size_util[int(i) + 1] = depth_left
                        i += 1
                    else:
                        mem_free[i] = capacity - mem_size_util[i]
                        ic_partition = mem_free[i] // (depth * oc * np.prod(ks))
                        mem_size_util[i + 1] = depth * (
                            node.weights - ic_partition * oc * np.prod(ks)
                        )
                        if mem_size_util[i + 1] > capacity:
                            depth_ratio = (
                                depth
                                * ((node.weights) - ic_partition * np.prod(ks) * oc)
                            ) // capacity

                            depth_left = (
                                depth
                                * ((node.weights) - ic_partition * np.prod(ks) * oc)
                            ) % capacity
                            i += int(depth_ratio)
                            message_passed[i + 1] = depth_ratio * (
                                np.prod(node.outputs[0].shape)
                                + np.prod(node.inputs[0].shape)
                            )

                            mem_size_util[int(i) + 1] = depth_left
                        i += 1
                    message_passed[i] += np.prod(node.outputs[0].shape) + np.prod(
                        node.inputs[0].shape
                    )

        # print(message_passed)
        print(np.sum(message_passed))


Mapper.run_asap = run_asap
Mapper.run_reuse_full = run_reuse_full
Mapper.run_reuse_leakage = run_reuse_leakage
Mapper.run_nn_dataflow = run_nn_dataflow
Mapper.illusion_mapping = illusion_mapping
