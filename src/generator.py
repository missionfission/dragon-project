import numpy as np
import pandas as pd
import yaml
import yamlordereddictloader

mem_table = np.array(pd.read_csv("tables/sram.csv", header=None))


# tech_table = np.array(pd.read_csv("tables/tech.csv"))

"""
Hyperparameters of Gradient Descent 
alpha
beta 
"""

alpha = 20000
beta = 4


class Generator:
    def __init__(self, constraintfiles="max.yaml"):
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


def backward_pass(self, scheduler, opts=None):

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
    """
    config = scheduler.config
    # config["compute"] = self.update_comp_design(scheduler, scheduler.config["compute"])
    config["memory"] = self.update_mem_design(scheduler, scheduler.config["memory"])

    return config


def backward_pass_tech(self, scheduler, opts=None):
    config = scheduler.config
    technology = config["technology"]
    technology = [
        technology["wire_cap"],
        technology["sense_amp_time"],
        technology["plogic_node"],
    ]
    mem_config = config["memory"]
    time_grads = (scheduler.mem_size_idle_time) / scheduler.total_cycles
    if opts == "time":
        technology = self.update_mem_tech("time", technology, time_grads=time_grads)

    if opts == "energy":
        mem_config["level0"]["banks"] += (int)(
            beta
            * scheduler.mem_energy
            / (np.sum(scheduler.mem_energy) + scheduler.compute_energy)
        )
        # Compute_energy is too high, check if compute is larger than required, or slower than required
        # compute_array size can be reduced -> how does compute array size effect energy consumption ->
        # it may be due to a lot of compute or bad-sized compute arrays

        # if mem energy consumption is too high at level 1, banks can be increased
        energy_grads = scheduler.mem_energy[0] / scheduler.total_energy
        technology = self.update_mem_tech(
            "read_energy", technology, energy_grads=energy_grads
        )
        technology = self.update_mem_tech(
            "write_energy", technology, energy_grads=energy_grads
        )

        ## What is really high read energy, write energy or leakage energy -> which depends on the leakage time
        # If leakage energy, read or write energy-> can change the technology type

        # if Energy is too high due to the leakage time : change sizing to energy efficient
        # If mem energy consumption is high -> which level ?
        # if mem_energy consumption is too high at level 0, its size can be reduced
    mem_config = mem_space(mem_config, technology)
    config["memory"] = mem_config
    return config


def update_comp_design(self, scheduler):
    pass


def update_mem_design(self, scheduler, mem_config):

    #  Allow changing for bandwidth and Size_idle_time -> bottlenecks always consume more time/energy
    # print("Bandwidth Idle Time", scheduler.bandwidth_idle_time)
    # print("Compute Idle Time", scheduler.compute_idle_time)
    # print("Memory Size Idle Time", scheduler.mem_size_idle_time)
    ## Sweep Connectivity : External bandwidth is sweeped : Bandwidth cannot be a bottleneck, say connectivity between 8 and 128
    # print(
    #     "Outside Memory Banks old",
    #     mem_config["level" + str(scheduler.mle - 1)]["banks"],
    # )

    if scheduler.bandwidth_idle_time > 0.1 * scheduler.total_cycles:
        if scheduler.force_connectivity is False:

            mem_config["level" + str(scheduler.mle - 1)]["banks"] += (int)(
                beta * scheduler.bandwidth_idle_time / scheduler.total_cycles
            )
        ## Force Connectivity : External bandwidth is forced, then cannot change anything
        ## Internal connectivity can change, memory banks can change
        if scheduler.internal_bandwidth_time > 0:
            mem_banks = mem_config["level1"]["banks"]
            mem_config["level1"]["banks"] += (
                beta * scheduler.bandwidth_idle_time / scheduler.total_cycles
            )
    # print(
    #     "Outside Memory Banks new",
    #     mem_config["level" + str(scheduler.mle - 1)]["banks"],
    # )

    ## If Mem Size idle time, Update mem size, update of size is proportional to the sizing of the memory
    # print("Memory Size old", mem_config["level0"]["size"])

    if scheduler.mem_size_idle_time > 0.1 * scheduler.total_cycles:
        mem_config["level0"]["size"] += (int)(
            alpha * scheduler.mem_size_idle_time / scheduler.total_cycles
        )
    # print("Memory Size new", mem_config["level0"]["size"])

    return mem_config


def update_mem_tech(self, opts, technology, time_grads=0, energy_grads=0):

    """
    opts is of either frequency or its for energy -> can modulate the access time of the cell and the cell energy
    opt = [frequency, read energy, write energy, leakage power, endurance]
    The technology space can be loaded from the file, and how we can rapidly find our point there
    The wire space is also loaded, and the joint technology and wire space can also be loaded
    """
    print("time_grads", time_grads)
    wire_cap, sense_amp_time, plogic_node = technology
    wire_cap = float(wire_cap)
    sense_amp_time = float(sense_amp_time)
    steps = 30
    ## We have to show that the memory cell space does not matter at all, all that matters is optimizing the wire space and the cmos space with it
    ## because above this interval it does not matter whether we can create a better technology or not.
    ## Joint sweep of tech space in cmos, memory cell and wires
    if opts == "energy" or opts == "read_energy":
        beta_wire = 1 / 50.7
        beta_sense_amp = 1 / 1.4
        beta_logic = 1
        # In the joint tech space that shows that sweeping wire space makes the real difference here
        wire_cap -= energy_grads * beta_wire
        plogic_node -= energy_grads * beta_logic

    if opts == "time":
        # In the joint time space that shows that sweeping cmos space makes the real difference
        # print("updating for time")
        beta_wire = 1 / 0.558
        beta_sense_amp = 1 / 1.4
        beta_logic = 1
        sense_amp_time -= steps * time_grads * beta_sense_amp
        wire_cap -= steps * time_grads * beta_wire

    print(wire_cap, sense_amp_time)
    return [wire_cap, sense_amp_time, plogic_node]


def mem_space(mem_config, technology):
    wire_cap, sense_amp_time, plogic_node = technology
    wire_cap = float(wire_cap)
    sense_amp_time = float(sense_amp_time)
    plogic_node = float(plogic_node)
    # mem_config["read_energy"] = (
    #     mem_config["level0"]["size"] * alpha_memory * (beta_cap + wire_cap * alpha_cap)
    #     + beta_read
    # )
    # mem_config["write_energy"] = (
    #     mem_config["level0"]["size"] * alpha_memory * (beta_cap + wire_cap * alpha_cap)
    #     + beta_write
    # )
    # mem_config["frequency"] = (
    #     mem_config["level0"]["size"] * alpha_memory * (beta_cap + wire_cap * alpha_cap)
    #     + beta_frequency
    # )
    mem_config["read_latency"] = 0.558 * wire_cap + 1.4 * sense_amp_time + 1.4
    mem_config["read_energy"] = 50.7 * wire_cap + 56.2
    mem_config["write_energy"] = 47.8 * wire_cap + 20
    mem_config["frequency"] = 1 / mem_config["read_latency"]
    return mem_config


def save_stats(self, scheduler, backprop=False, backprop_memory=0):
    """
    Execution statistics also have to be generated : Area, Energy, Time/Number of Cycles 
    """
    config = scheduler.config
    mem_config = config["memory"]
    mm_compute = config["mm_compute"]
    # mem_area = np.zeros((scheduler.mle))
    # compute_area = (
    #     self.get_compute_area(mm_compute["class"], mm_compute["size"])
    #     * mm_compute["N_PE"]
    # )
    total_energy = 0
    mem_energy = np.zeros((scheduler.mle))
    compute_energy = (
        mm_compute["N_PE"]
        * mm_compute["size"]
        * (
            scheduler.total_cycles
            - scheduler.bandwidth_idle_time
            - scheduler.mem_size_idle_time
        )
        * 12.75
    ) / 1000
    for i in range(scheduler.mle - 1):
        memory = config["memory"]["level" + str(i)]
        read_energy = float(memory["read_energy"])
        write_energy = float(memory["write_energy"])
        leakage_power = float(memory["leakage_power"])
        mem_energy[i] = (
            scheduler.mem_read_access[i] * read_energy
            + scheduler.mem_write_access[i] * write_energy
            + leakage_power * scheduler.total_cycles
        ) / 100
        # print(read_energy, write_energy, leakage_power)
        # print(mem_energy)
    mem_energy[scheduler.mle - 1] = (
        scheduler.mem_read_access[i] + scheduler.mem_write_access[i]
    ) / 5000

    # total_area = np.sum(mem_area) + compute_area
    total_energy = np.sum(mem_energy) + compute_energy
    scheduler.mem_energy = mem_energy
    scheduler.conpute_energy = compute_energy
    scheduler.logger.info("Tool Output")
    scheduler.logger.info("===========================")
    scheduler.logger.info("Total No of cycles  = %d ", scheduler.total_cycles)
    scheduler.logger.info("Memory Energy Consumption  = %d ", np.sum(mem_energy))
    scheduler.logger.info("Compute Energy Consumption  = %d ", compute_energy)
    scheduler.logger.info(" Total Energy Consumption  = %d ", total_energy)
    # scheduler.logger.info("Memory Area Consumption  = %d ", np.sum(mem_area))
    # scheduler.logger.info("Compute Area Consumption  = %d ", compute_area)
    # scheduler.logger.info("Total Area Consumption  = %d ", total_area)
    if backprop:
        scheduler.total_cycles = (
            2 * scheduler.total_cycles
            + backprop_memory // scheduler.mem_read_bw[scheduler.mle - 1]
        )
        scheduler.bandwidth_idle_time += (
            backprop_memory // scheduler.mem_read_bw[scheduler.mle - 1]
        )
    config = scheduler.config
    technology = config["technology"]
    technology = [
        float(technology["wire_cap"]),
        float(technology["sense_amp_time"]),
        float(technology["plogic_node"]),
    ]
    print("==================================")
    print(
        "Time",
        int(scheduler.total_cycles),
        int(scheduler.bandwidth_idle_time),
        scheduler.compute_idle_time,
        int(scheduler.mem_size_idle_time),
    )
    print(
        "Energy",
        int(total_energy),
        int(compute_energy),
        int(scheduler.mem_read_access[0] * read_energy / 100),
        int(scheduler.mem_write_access[0] * write_energy / 100),
        int(leakage_power * scheduler.total_cycles / 1000),
        int(scheduler.mem_read_access[1] / 5000),
        int(scheduler.mem_write_access[1] / 5000),
        int(1 * scheduler.total_cycles),
    )
    print(
        "Design Params",
        mem_config["level" + str(scheduler.mle - 1)]["banks"],
        mem_config["level0"]["size"],
    )

    print("Tech Params", technology)

    assert scheduler.total_cycles > scheduler.bandwidth_idle_time
    assert scheduler.total_cycles > scheduler.mem_size_idle_time
    assert scheduler.bandwidth_idle_time > 0
    # assert scheduler.mem_size_idle_time > 0
    return (
        [
            int(scheduler.total_cycles),
            int(scheduler.bandwidth_idle_time),
            int(scheduler.mem_size_idle_time),  #
            scheduler.compute_idle_time,
        ],
        [
            int(total_energy),
            int(compute_energy),
            int(scheduler.mem_read_access[0] * read_energy / 100),
            int(scheduler.mem_write_access[0] * write_energy / 100),
            int(leakage_power * scheduler.total_cycles / 1000),
            int(scheduler.mem_read_access[1] / 5000),
            int(scheduler.mem_write_access[1] / 5000),
            int(1 * scheduler.total_cycles),
        ],
        [
            mem_config["level" + str(scheduler.mle - 1)]["banks"],
            mem_config["level0"]["size"],
        ],
        technology,
    )


#############################################################################################################################


def get_compute_props(*args, **kwargs):
    pass


def get_mem_props(size, width, banks):
    for i in range(11, 25):
        if (size // 2 ** i) < 1:
            break
    a = mem_table[np.where(mem_table[:, 1] == banks)]
    a = a[np.where(a[:, 2] == width)]
    element = min(a[:, 0], key=lambda x: abs(x - size))
    a = a[np.where(a[:, 0] == element)]
    return a[0, 5], a[0, 6], a[0, 7]


def analyze3d(self, opts):
    """
    Are Values Realistic : is the value of bandwidth and number of connections etc < maxval
    Is the size of memory array < maxval
    Check if noc time is ever going to be a problem ?
    Check whether minval and maxval satisfy here ?
    """
    pass


def endurance_writes_schedule():
    pass
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


Generator.save_stats = save_stats
Generator.update_mem_tech = update_mem_tech
Generator.update_comp_design = update_comp_design
Generator.update_mem_design = update_mem_design
Generator.backward_pass = backward_pass
Generator.backward_pass_tech = backward_pass_tech
Generator.writeconfig = writeconfig
Generator.analyze3d = analyze3d
