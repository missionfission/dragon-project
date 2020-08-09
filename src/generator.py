import numpy as np
import pandas as pd
import yaml
import yamlordereddictloader

mem_table = np.array(pd.read_csv("tables/sram.csv"))
# compute_table = np.array(pd.read_csv("tables/compute.csv"))

"""

Hyperparameters of Gradient Descent 

alpha


beta 


"""


class Generator:
    def __init__(self, constraintfiles="max.yaml"):
        base_dir = "configs/"

        self.maxval = yaml.load(
            open(base_dir + constraintfiles), Loader=yamlordereddictloader.Loader
        )


def writehwfile(self, content, filename):
    """
    Generate Hardware Description Yaml File 
    """
    outfile = open(filename, "w")
    outfile.write(
        yaml.dump(
            content, default_flow_style=False, Dumper=yamlordereddictloader.SafeDumper,
        )
    )


def findnext(self, scheduler, opts=None):

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
    newhw = scheduler.config
    print(
        scheduler.bandwidth_idle_time,
        scheduler.compute_idle_time,
        scheduler.mem_size_idle_time,
    )
    self.updatecomputedesign()
    self.updatememdesign()

    if opts == "time":
        self.findmemtechnology("frequency")

    if opts == "energy":
        print(scheduler.mem_energy, scheduler.compute_energy)
        # Compute_energy is too high, check if compute is larger than required, or slower than required
        # compute_array size can be reduced -> how does compute array size effect energy consumption ->
        # it may be due to a lot of compute or bad-sized compute arrays

        # if mem energy consumption is too high at level 1, bandwidth->frequency can be reduced
        self.findmemtechnology("read_energy")

        ## What is really high read energy, write energy or leakage energy -> which depends on the leakage time
        # If leakage energy, read or write energy-> can change the technology type

        # if Energy is too high due to the leakage time : change sizing to energy efficient
        # If mem energy consumption is high -> which level ?
        # if mem_energy consumption is too high at level 0, its size can be reduced

    return newhw


def get_mem_energy(self, size, read_ports, banks):
    return mem_table[i]


def updatecomputedesign(self, scheduler):
    pass


def updatememdesign(self, scheduler, mem_config):

    #  Allow changing for bandwidth and Size_idle_time -> bottlenecks always consume more time/energy

    ## Sweep Connectivity : External bandwidth is sweeped : Bandwidth cannot be a bottleneck, say connectivity between 8 and 128
    if scheduler.bandwidth_idle_time > 0.1 * scheduler.total_time:
        if scheduler.force_connectivity is False:
            mem_banks = mem_config["level" + str(scheduler.mle - 1)]["banks"]
            mem_size = mem_config["level0"]["size"]
            mem_banks += beta * scheduler.bandwidth_idle_time / scheduler.total_time
        ## Force Connectivity : External bandwidth is forced, then cannot change anything
        ## Internal connectivity can change, memory banks can change
        if scheduler.internal_bandwidth_time > 0:
            mem_banks = mem_config["level1"]["banks"]
            mem_banks += beta * scheduler.bandwidth_idle_time / scheduler.total_time

    ## If Mem Size idle time, Update mem size, update of size is proportional to the sizing of the memory
    if scheduler.size_idle_time > 0.1 * scheduler.total_time:
        mem_size += alpha * scheduler.size_idle_time / scheduler.total_time


def findmemtechnology(self, opts, time_grads, energy_grads):

    """
    opts is of either frequency or its for energy -> can modulate the access time of the cell and the cell energy
    opt = [frequency, read energy, write energy, leakage power, endurance]
    The technology space can be loaded from the file, and how we can rapidly find our point there
    The wire space is also loaded, and the joint technology and wire space can also be loaded
    """
    ## For joint optimization of technology space and wire space, we produce the sensitivity analysis,
    ## of technology space and wire space and how it affects the design space in energy of access
    ## It can affect the frequency in the sizing of the memory arrays, it affect the energy of accesses also

    ## Figure out sensitivity from the differentiable models, how much does tech and wire parameters affect the design space.


def save_statistics(self, scheduler):
    """
    Execution statistics also have to be generated : Area, Energy, Time/Number of Cycles 
    """
    config = scheduler.config
    mm_compute = config["mm_compute"]
    mem_area = np.zeros((scheduler.mle))

    # compute_area = (
    #     self.get_compute_area(mm_compute["class"], mm_compute["size"])
    #     * mm_compute["N_PE"]
    # )

    mem_energy_access = np.zeros((scheduler.mle, 2))
    mem_energy = np.zeros((scheduler.mle))
    compute_energy = macs * unit_energy

    for i in range(scheduler.mle):
        memory = config["memory"]["level" + str(i)]
        mem_energy_access[i] = mem_access[i] * self.get_mem_energy(
            memory["size"], memory["read_ports"], memory["banks"]
        )
        # mem_area[i] = self.get_mem_area(
        # memory["size"], memory["read_ports"], memory["banks"], connectivity,
        # )

    # total_area = np.sum(mem_area) + compute_area
    total_energy = np.sum(mem_energy) + compute_energy

    scheduler.logger.info("Tool Output")
    scheduler.logger.info("===========================")
    scheduler.logger.info("Total No of cycles  = %d ", scheduler.total_cycles)
    scheduler.logger.info("Memory Energy Consumption  = %d ", np.sum(mem_energy))
    scheduler.logger.info("Compute Energy Consumption  = %d ", compute_energy)
    scheduler.logger.info(" Total Energy Consumption  = %d ", total_energy)
    # scheduler.logger.info("Memory Area Consumption  = %d ", np.sum(mem_area))
    # scheduler.logger.info("Compute Area Consumption  = %d ", compute_area)
    # scheduler.logger.info("Total Area Consumption  = %d ", total_area)


#############################################################################################################################


def get_compute_energy(self, *args, **kwargs):
    pass


def get_compute_area(self, *args, **kwargs):
    pass


def get_mem_area(self, *args, **kwargs):
    pass


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


Generator.save_statistics = save_statistics
Generator.findmemtechnology = findmemtechnology
Generator.updatecomputedesign = updatecomputedesign
Generator.updatememdesign = updatememdesign
Generator.findnext = findnext
Generator.writehwfile = writehwfile
Generator.get_mem_energy = get_mem_energy
Generator.get_compute_energy = get_compute_energy
Generator.get_compute_area = get_compute_area
Generator.get_mem_area = get_mem_area
Generator.analyze3d = analyze3d
