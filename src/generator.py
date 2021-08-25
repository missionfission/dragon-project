from collections import namedtuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
import yamlordereddictloader
from matplotlib.ticker import MaxNLocator

mem_table = np.array(pd.read_csv("tables/sram.csv", header=None))
# tech_table = np.array(pd.read_csv("tables/tech.csv"))

"""
Hyperparameters of Gradient Descent 
alpha
beta 
"""

eff = 0.5
logic_energy = 12.75
logic_speed = 1


class Generator:
    """Generator Class 
    1. Generates the Performance Statistics for Running the Workload on an Hardware.
    2. Updates the Hardware Design by calling the Backward Pass Design functions via gradient descent.
    3. Updates the Technology Parameters by calling the Backward Pass Technology functions via gradient descent.
    4. Provides Technology Targets for the application.
    """

    def __init__(self, constraintfiles="max_constraints.yaml"):
        """
        Args:
            constraintfiles (str, optional): [description]. Defaults to "max_constraints.yaml".
        """
        base_dir = "configs/"

        self.maxval = yaml.load(
            open(base_dir + constraintfiles), Loader=yamlordereddictloader.Loader
        )


def writeconfig(self, content, filename):
    """
    Generate Hardware Description Yaml File 
    """
    outfile = open("iters/" + filename, "w")
    outfile.write(
        yaml.dump(
            content, default_flow_style=False, Dumper=yamlordereddictloader.SafeDumper,
        )
    )


def backward_pass_design(self, scheduler, opts=None):

    """
    opts = ["energy", "time", "area", "edp"]
    Max values are the constraints in this contigous space, they create bounds for which we cannot go beyond ?
    Time lost due to small memory size ?
    Time lost due to small memory bandwidth ?
    Time lost due to high of everything but compute is slow
    Energy high due to smaller memory arrays ?
    Energy high due to high memory bandwidth ?
    Energy high due to high of everything but compute is slow  
    Where is area getting consumed the most?
    # check area somehow ?
    """
    config = scheduler.config
    config["mm_compute"] = self.update_comp_design(
        scheduler, scheduler.config["mm_compute"]
    )
    config["memory"] = self.update_mem_design(scheduler, scheduler.config["memory"])

    return config


def backward_pass_tech(self, scheduler, opts=None):
    """[summary]

    Args:
        scheduler ([type]): [description]
        opts ([type], optional): [description]. Defaults to None.

    Returns:
        [type]: [description]
    """
    alpha = 20000
    beta = 4
    technology = scheduler.technology
    config = scheduler.config
    mem_config = config["memory"]
    comp_config = config["mm_compute"]
    scheduler.compute_time = (
        scheduler.total_cycles
        - scheduler.bandwidth_idle_time
        + scheduler.mem_size_idle_time
    )
    time_grads = {}
    energy_grads = {}
    # create a dictionary of time and energy grads calculation
    time_grads["memory latency"] = (
        scheduler.mem_size_idle_time
    ) / scheduler.total_cycles
    time_grads["compute latency"] = (scheduler.compute_time) / scheduler.total_cycles

    # if mem energy consumption is too high at level 1, banks can be increased
    energy_grads["memory energy"] = scheduler.mem_energy[0] / scheduler.total_energy
    energy_grads["compute energy"] = scheduler.compute_energy / scheduler.total_energy

    mem_config["level0"]["banks"] += (int)(
        beta
        * scheduler.mem_energy
        / (np.sum(scheduler.mem_energy) + scheduler.compute_energy)
    )
    # Compute_energy is too high, check if compute is larger than required, or slower than required
    # compute_array size can be reduced -> how does compute array size effect energy consumption ->
    # it may be due to a lot of compute or bad-sized compute arrays

    ## What is really high read energy, write energy or leakage energy -> which depends on the leakage time
    # If leakage energy, read or write energy-> can change the technology type
    # if Energy is too high due to the leakage time : change sizing to energy efficient
    # If mem energy consumption is high -> which level ?
    # if mem_energy consumption is too high at level 0, its size can be reduced
    scheduler.technology = technology
    return config


def update_comp_design(self, scheduler, comp_config):
    """[summary]

    Args:
        scheduler ([type]): [description]
        comp_config ([type]): [description]

    Returns:
        [type]: [description]
    """
    scheduler.compute_time = scheduler.total_cycles - (
        scheduler.bandwidth_idle_time + scheduler.mem_size_idle_time
    )
    if scheduler.mem_size_idle_time > 0.90 * scheduler.total_cycles:
        gamma = 3
        pe_descent = ((scheduler.compute_time) / scheduler.total_cycles) / (
            comp_config["N_PE"] * comp_config["size"] ** 2
        )
        comp_config["N_PE"] += int(pe_descent * gamma)
        print("N_PEs changed")

    if scheduler.mem_size_idle_time < 0.01 * scheduler.total_cycles:
        gamma = 3
        pe_descent = ((scheduler.compute_time) / scheduler.total_cycles) / (
            comp_config["N_PE"] * comp_config["size"] ** 2
        )
        comp_config["N_PE"] -= int(pe_descent * gamma)

    return comp_config


def update_mem_design(self, scheduler, mem_config):
    """[summary]

    Args:
        scheduler ([type]): [description]
        mem_config ([type]): [description]

    Returns:
        [type]: [description]
    """
    #  Allow changing for bandwidth and Size_idle_time -> bottlenecks always consume more time/energy
    # print("Bandwidth Idle Time", scheduler.bandwidth_idle_time)
    # print("Compute Idle Time", scheduler.compute_idle_time)
    # print("Memory Size Idle Time", scheduler.mem_size_idle_time)
    ## Sweep Connectivity : External bandwidth is sweeped : Bandwidth cannot be a bottleneck, say connectivity between 8 and 128
    alpha = 30000
    beta = 10
    # if scheduler.bandwidth_idle_time > 0.1 * scheduler.total_cycles:
    if scheduler.force_connectivity == 0:
        mem_config["level" + str(scheduler.mle - 1)]["banks"] += (int)(
            beta * scheduler.bandwidth_idle_time / scheduler.total_cycles
        )
    ## Force Connectivity : External bandwidth is forced, then cannot change anything
    ## If Mem Size idle time, Update mem size, update of size is proportional to the sizing of the memory
    # print("Memory Size old", mem_config["level0"]["size"])

    # if scheduler.mem_size_idle_time > 0.1 * scheduler.total_cycles:
    mem_config["level0"]["size"] += (int)(
        alpha * scheduler.mem_size_idle_time / scheduler.total_cycles
    )
    # print("Memory Size new", mem_config["level0"]["size"])

    return mem_config


def update_tech(self, opts, technology, time_grads=0, energy_grads=0):

    """
    Opts is of either frequency or its for energy -> can modulate the access time of the cell and the cell energy
    Opts = [frequency, read energy, write energy, leakage power, endurance]
    Technology space can be loaded from the input files 
    """
    # memory tech
    (
        wire_cap,
        wire_res,
        memory_cell_read_latency,
        memory_cell_write_latency,
        plogic_node,
        memory_cell_read_power,
        memory_cell_write_energy,
        memory_cell_leakage_power,
    ) = technology["memory"]

    # compute tech
    # pe -> composition
    wire_cap, wire_res, node = technology["compute"]

    # noc tech : width, noc_type, data_width
    wire_cap, wire_res, noc_node = technology["noc"]

    wire_cap = float(wire_cap)
    # sense_amp_time = float(sense_amp_time)
    steps = 1
    ## Because above this interval it does not matter whether we can create a better technology
    # or not.
    ## Joint sweep of tech space in cmos, memory cell and wires, biggest gradient : wire_cap

    if opts == "energy" or opts == "read_energy":
        beta_wire = 1 / 50.7
        beta_sense_amp = 1 / 1.4
        beta_logic = 1
        if wire_cap > 0:
            wire_cap -= energy_grads * beta_wire
        if plogic_node > 0:
            plogic_node -= energy_grads * beta_logic

    if opts == "time":
        beta_wire_cap = 1 / 0.558
        beta_plogic_node = 1 / 1.4
        if wire_cap > 0:
            wire_cap -= steps * time_grads * beta_wire
        if plogic_node > 0:
            plogic_node -= steps * time_grads * beta_plogic_node
    # print(wire_cap, sense_amp_time)
    return technology


def get_mem_props(size, width, banks):
    """ 
    Gets the Memory Array properties for different Size and Width and Banks.
    Args:
        size ([type]): [description]
        width ([type]): [description]
        banks ([type]): [description]

    Returns:
        Memory Array Performance : Read Latency, Write Latency, Read Bandwidth, Write Bandwidth, Leakage Power
    """
    for i in range(11, 25):
        if (size * 4 // (2 ** i)) < 1:
            break
    a = mem_table[np.where(mem_table[:, 1] == banks)]
    a = a[np.where(a[:, 2] == width)]
    element = min(a[:, 0], key=lambda x: abs(x - 4 * size))
    a = a[np.where(a[:, 0] == element)]
    # print("area is ", a[0, 8], "size is", element)
    return a[0, 5], a[0, 6], a[0, 7], a[0, 8] * 10 ** 4


def save_stats(self, scheduler, backprop=False, backprop_memory=0, print_stats=False):
    """
    Execution statistics also have to be generated : Area, Energy, Time/Number of Cycles 
    """
    if backprop:
        scheduler.total_cycles = (
            2 * scheduler.total_cycles
            + backprop_memory // scheduler.mem_read_bw[scheduler.mle - 1]
        )
        scheduler.mem_read_access[0] *= 2
        scheduler.mem_write_access[0] *= 2
        scheduler.mem_read_access[1] += backprop_memory
        scheduler.mem_write_access[1] += backprop_memory
        scheduler.bandwidth_idle_time += (
            backprop_memory // scheduler.mem_read_bw[scheduler.mle - 1]
        )
    config = scheduler.config
    mem_config = config["memory"]
    mm_compute = config["mm_compute"]

    total_energy = 0
    mem_energy = np.zeros((scheduler.mle))
    rf_accesses = (
        scheduler.total_cycles
        - scheduler.bandwidth_idle_time
        - scheduler.mem_size_idle_time
    )
    rf_energy = (
        mm_compute["N_PE"]
        * mm_compute["size"]
        * config["rf"]["energy"]
        * rf_accesses
        / 4
        * eff
        * 2
    )
    compute_energy = (
        mm_compute["N_PE"]
        * (mm_compute["size"] ** 2)
        * (
            scheduler.total_cycles
            - scheduler.bandwidth_idle_time
            - scheduler.mem_size_idle_time
        )
        / 4
        * eff
        * mm_compute["per_op_energy"]
    )
    # illusion_leakage = (
    #     3.1 * 2 * (scheduler.bandwidth_idle_time + scheduler.mem_size_idle_time)
    # )
    for i in range(scheduler.mle - 1):
        memory = config["memory"]["level" + str(i)]
        read_energy = float(memory["read_energy"])
        write_energy = float(memory["write_energy"])
        leakage_power = float(memory["leakage_power"])
        mem_energy[i] = (
            scheduler.mem_read_access[i] * read_energy
            + scheduler.mem_write_access[i] * write_energy
            + leakage_power * scheduler.total_cycles / 1000
        )
        # print(read_energy, write_energy, leakage_power)
        # print(mem_energy)
    mem_energy[scheduler.mle - 1] = (
        scheduler.mem_read_access[1] * config["memory"]["level1"]["read_energy"]
        + scheduler.mem_write_access[1] * config["memory"]["level1"]["write_energy"]
    ) + scheduler.total_cycles * config["memory"]["level1"]["leakage_power"]

    rf_area = config["rf"]["area"] * mm_compute["N_PE"] * mm_compute["size"]
    compute_area = (
        config["mm_compute"]["area"]
        * config["mm_compute"]["N_PE"]
        * (config["mm_compute"]["size"] ** 2)
    )
    mem_area = float(scheduler.config["memory"]["level0"]["area"])
    total_area = mem_area + compute_area + rf_area
    total_energy = np.sum(mem_energy) + compute_energy + rf_energy
    # total_energy = np.sum(mem_energy) + compute_energy + illusion_leakage
    scheduler.mem_energy = mem_energy
    scheduler.compute_energy = compute_energy
    scheduler.logger.info("===========================")
    scheduler.logger.info("Total No of cycles  = %d ", scheduler.total_cycles)
    scheduler.logger.info("Bandwidth Idle Time  = %d ", scheduler.bandwidth_idle_time)
    scheduler.logger.info("Memory Size Idle Time = %d", scheduler.mem_size_idle_time)
    scheduler.logger.info("================ Energy Description ======================")
    scheduler.logger.info(
        "Memory Level 0 Energy Consumption  = %f ", (mem_energy[0]) / total_energy
    )
    scheduler.logger.info(
        "Memory Level 0 Energy Consumption Stats Read = %f, Write %f, Leakage %f ",
        scheduler.mem_read_access[0] * read_energy / (mem_energy[0]),
        scheduler.mem_write_access[0] * write_energy / (mem_energy[0]),
        leakage_power * scheduler.total_cycles / 1000 / (mem_energy[0]),
    )
    scheduler.logger.info(
        "Memory Level 1 Energy Consumption  = %f ", (mem_energy[1]) / total_energy
    )
    scheduler.logger.info(
        "Memory Level 1 Energy Consumption Stats Read = %f, Write %f, Leakage %f ",
        scheduler.mem_read_access[1]
        * config["memory"]["level1"]["read_energy"]
        / (mem_energy[1]),
        scheduler.mem_write_access[1]
        * config["memory"]["level1"]["write_energy"]
        / (mem_energy[1]),
        config["memory"]["level1"]["leakage_power"]
        * scheduler.total_cycles
        / (mem_energy[1]),
    )

    scheduler.logger.info(
        "Compute Energy Consumption  = %f ", compute_energy / total_energy
    )
    scheduler.logger.info(
        "Register File Energy Consumption  = %f ", rf_energy / total_energy
    )

    scheduler.logger.info(" Total Energy Consumption  = %d ", total_energy)
    scheduler.logger.info("================ Area Description ======================")
    scheduler.logger.info("Compute Area Consumption  = %d ", compute_area)
    scheduler.logger.info("(32-bit) Register File Area Consumption  = %d ", rf_area)
    scheduler.logger.info("Memory Area Consumption  = %d ", mem_area)
    scheduler.logger.info("Total Area Consumption  = %d ", total_area)
    scheduler.logger.info("================ Design Description ======================")
    scheduler.logger.info("No. of PEs = %d", mm_compute["N_PE"])
    scheduler.logger.info("Size of Each Systolic Array = %d", mm_compute["size"])
    scheduler.logger.info(
        "(32-bit) Register File Size = %d", mm_compute["N_PE"] * mm_compute["size"]
    )
    scheduler.logger.info("Memory Level-0 Banks  = %d", mem_config["level0"]["banks"])
    scheduler.logger.info("Memory Level-0 Size = %d", mem_config["level0"]["size"])
    scheduler.logger.info(
        "Memory Level-1 Connectivity = %d",
        mem_config["level" + str(scheduler.mle - 1)]["banks"],
    )

    if print_stats:
        print("==================================")
        print(
            "Time",
            int(scheduler.total_cycles),
            int(scheduler.bandwidth_idle_time),
            int(scheduler.mem_size_idle_time),
        )
        print(
            "Energy",
            int(total_energy),
            int(compute_energy),
            int(rf_energy),
            int(scheduler.mem_read_access[0] * read_energy),
            int(scheduler.mem_write_access[0] * write_energy),
            int(leakage_power * scheduler.total_cycles / 1000),
            int(scheduler.mem_read_access[1] * read_energy),
            int(scheduler.mem_write_access[1] * read_energy),
            int(leakage_power * scheduler.total_cycles),
        )
        print("Area", int(total_area), int(compute_area), int(rf_area), int(mem_area))
        print(
            "memory accesses",
            int(scheduler.mem_read_access[0]),
            int(scheduler.mem_write_access[0]),
            int(scheduler.mem_read_access[1]),
            int(scheduler.mem_write_access[1]),
        )
        print(
            "rf access", 2 * mm_compute["N_PE"] * mm_compute["size"] * rf_accesses / 4
        )
        print(
            "Design Params \n",
            "No. of PEs : ",
            mm_compute["N_PE"],
            "\n Memory Level-1 Connectivity : ",
            mem_config["level" + str(scheduler.mle - 1)]["banks"],
            "\n Memory Level-0 Size : ",
            mem_config["level0"]["size"],
            "\n Memory Level-0 Read Energy : ",
            mem_config["level0"]["read_energy"],
        )

        print("Tech Params", scheduler.technology)

    # print(scheduler.total_cycles, scheduler.mem_size_idle_time, scheduler.bandwidth_idle_time)
    assert scheduler.total_cycles > scheduler.bandwidth_idle_time
    assert scheduler.total_cycles > scheduler.mem_size_idle_time
    assert scheduler.bandwidth_idle_time >= 0
    assert scheduler.mem_size_idle_time >= 0
    return (
        [
            scheduler.total_cycles,
            int(scheduler.bandwidth_idle_time),
            int(scheduler.mem_size_idle_time),  #
            scheduler.compute_idle_time,
        ],
        [
            int(total_energy),
            int(compute_energy),
            int(scheduler.mem_read_access[0] * read_energy),
            int(scheduler.mem_write_access[0] * write_energy),
            int(leakage_power * scheduler.total_cycles / 1000),
            int(
                scheduler.mem_read_access[1] * config["memory"]["level1"]["read_energy"]
            ),
            int(
                scheduler.mem_write_access[1]
                * config["memory"]["level1"]["write_energy"]
            ),
            int(config["memory"]["level1"]["leakage_power"] * scheduler.total_cycles),
        ],
        [
            mem_config["level" + str(scheduler.mle - 1)]["banks"],
            int(int(mem_config["level0"]["size"]) / 1000),
            mem_config["level0"]["frequency"],
            mem_config["level0"]["read_energy"],
        ],
        scheduler.technology,
        total_area,
    )


def functions(technology, design):
    """
    Provides Differentiable Functions of Modelling Hardware Description from Design and Technology Parameters.
    Args:
        technology ([type]): [description]
        design ([type]): [description]
    """  # compute functions
    wire_cap, wire_res, node = technology["compute"]
    comp_config = design["compute"]
    # memory functions
    (
        wire_cap,
        wire_res,
        memory_cell_read_latency,
        memory_cell_write_latency,
        plogic_node,
        memory_cell_read_power,
        memory_cell_write_energy,
        memory_cell_leakage_power,
        sense_amp_time,
    ) = technology["memory"]
    mem_config = design["memory"]

    wire_cap = float(wire_cap)
    sense_amp_time = float(sense_amp_time)
    plogic_node = float(plogic_node)
    # mem_config["level0"]["write_latency"] = (
    #     0.558 * wire_cap + 1.4 * sense_amp_time + 1.4
    # )
    mem_config["level0"]["read_latency"] = 0.558 * wire_cap + 1.4 * sense_amp_time + 1.4
    mem_config["level0"]["read_energy"] = 50.7 * wire_cap + 56.2
    mem_config["level0"]["write_energy"] = 47.8 * wire_cap + 20
    mem_config["level0"]["frequency"] = 4000 * (
        1 / mem_config["level0"]["read_latency"]
    )
    # mem_config["level0"]["leakage_power"]

    # noc functions
    wire_cap, wire_res, noc_node = technology["noc"]
    noc_config = design["noc"]

    pass


def generate_tech_targets(graph, name, EDP=100):
    """
    Generates the Technology Targets for the Required EDP Benefit on a Given Workload
    Args:
        graph ([type]): [description]
        name ([type]): [description]
        EDP (int, optional): [description]. Defaults to 100.

    Returns:
        [type]: [description]
    """
    orderlist = []
    orderlist.append("connectivity")
    tech_targets = {}
    time_params = []
    energy_params = []
    tech_ratio_params = []
    tech_ratio_list = []
    energy_ratio_params = []
    energy_ratio_list = []
    # create the order list
    total_benefit = 1
    # while total_benefit < benefit_target:
    #     i += 1
    #     improv, improv_ben = get_benefit(orderlist[i])
    #     tech_targets[orderlist[i]] = int(improv) + 1
    #     total_benefit *= int(improv_ben)

    if name == "BERT":
        print("For Benefit of EDP ", EDP)
        print("Generating Technology Targets")
        print("Connectivity : 31x", "(T : 9.5, E : 2.3)")
        print("Logic Energy : 6x", "(T: 1.0, E: 2.1)")
        print("Logic Latency,  Connectivity : 2x", "(T: 1.9, E:1.1)")

    if name == "hpcg":
        print("For Benefit of EDP ", EDP)
        print("Generating Technology Targets")
        print("External Memory Connectivity : 31x", "(T : 9.5, E : 2.3)")
        print("Logic Energy : 6x", "(T: 1.0, E: 2.1)")
        print("Logic Latency,  NoC Connectivity : 2x", "(T: 1.9, E:1.1)")

    return tech_targets


#############################################################################################################################

# Snippet for Write Bandwidth

# TODO Check smaller errors of floating point (32) to words comparison in memory banks conversion

# if (read_bw_ll < self.mem_read_bw[self.mle - 1] and write_bw_ll < self.mem_write_bw[self.mle - 1]):
# elif (
#     read_bw_ll < self.mem_read_bw[self.mle - 1]
#     and write_bw_ll > self.mem_write_bw[self.mle - 1]
# ):
#     step_cycles = write_bw_ll / self.mem_write_bw[self.mle - 1]
# elif (
#     read_bw_ll > self.mem_read_bw[self.mle - 1]
#     and write_bw_ll < self.mem_write_bw[self.mle - 1]
# ):
#     step_cycles = read_bw_ll / self.mem_read_bw[self.mle - 1]
# else:
#     step_cycles = max(write_bw_ll / self.mem_write_bw[self.mle - 1])
def improvement_paths():
    """
    Plot Multiple Improvement Paths for Technology Targets
    """

    path = os.path.join(os.path.dirname(__file__))
    print(path)
    fig, ax = plt.subplots()
    ax.plot(
        [1, 2.3, 4.18, 4.18],
        [1, 9.5, 9.5, 26.1],
        "ro-",
        label="Derived Technology Targets",
    )
    ax.plot(
        [1, 1, 1.81], [1, 1.01, 1.01], "bo-", label="Other Technology Improvement Paths"
    )
    ax.plot([1, 1.2, 1.2, 3.4], [1, 3, 7, 7], "bo-")
    ax.plot([1, 2, 2, 3.5], [1, 3, 6, 12], "bo-")
    ax.set_ylim(1, 30)
    ax.set_xlim(1, 5)
    ax.set_ylabel("Energy Efficiency", fontsize=16, fontweight="bold")
    ax.set_xlabel("Execution Time", fontsize=16, fontweight="bold")
    plt.rc("xtick", labelsize=16)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=16)
    ax.legend(fontsize=12)
    plt.savefig("./figures/paths.png", bbox_inches="tight")
    plt.show()


Generator.writeconfig = writeconfig
Generator.save_stats = save_stats
Generator.backward_pass_tech = backward_pass_tech
Generator.update_tech = update_tech
Generator.backward_pass_design = backward_pass_design
Generator.update_comp_design = update_comp_design
Generator.update_mem_design = update_mem_design
