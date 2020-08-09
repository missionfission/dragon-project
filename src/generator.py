import numpy as np
import pandas as pd
import yaml
import yamlordereddictloader

mem_table = np.array(pd.read_csv("tables/sram.csv"))
# compute_table = np.array(pd.read_csv("tables/compute.csv"))


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
                content,
                default_flow_style=False,
                Dumper=yamlordereddictloader.SafeDumper,
            )
        )

    def findnext(self, scheduler, opts=None):
        # opts = ["energy", "time", "area", "edp"]
        """
        We parameterize the contigous space of timing, area and memory of the various technologies 
        So, several technologies appear as different curves in this space : right ?
        So, to find a better point in this multidimensional contigous space of technologies and hardware,
        we can analyze the effects of various bottlenecks in the data-flow graph processing : Then we can do what wasn't possible 
        The trade-offs of memory sizing and technology sizing are clearly visible in this space.
        Then we combine this space, total architecture is the joint space of memory, compute and networking space 
        Now the combination of this space is a space -> Convex Optimization 

        Say, a memory bottleneck is encountered in the data-flow graph : that changes this contigous space, 
        we also get a data-flow graph space ?, and from that data-flow graph space, we can get a new point of graphical representation ?



        Max values are the constraints in this contigous space, they create bounds for which we cannot go beyond ?

        compute_idle_time
        increase_memory_size
        increase_memory_bandwidth/memory_connections
        increase number of compute_arrays
        change size of compute_arrays
        change number of memory banks
        Are Values Realistic : is the value of bandwidth and number of connections etc < maxval
        Is the size of memory array < maxval
        Check if noc time is ever going to be a problem ?
        Check whether minval and maxval satisfy here ?

        Time lost due to small memory size ?
        Time lost due to small memory bandwidth ?
        Time lost due to high of everything but compute is slow
        Energy high due to small memory size ?
        Energy high due to small memory bandwidth ?
        Energy high due to high of everything but compute is slow
        
        Where is area getting consumed the most?
        """
        ## Sweep Connectivity : External bandwidth is sweeped : Bandwidth cannot be a bottleneck

        ## Force Connectivity : External bandwidth is forced

        if opts == "energy":
            pass
            #  Allow changing for bandwidth and Size_idle_time -> bottlenecks always consume more energy

            # if Energy is too high due to the leakage time :

            # Compute_energy is too high, check if compute is larger than required,
            # compute_array size can be reduced -> how does compute array size effect energy consumption -> it may also increase compute bottleneck

            # If mem energy consumption is high -> which level ?, is there a lot of prefetching in terms of compute_idle_time or
            # underutilized bandwidth/size

            # if mem_energy consumption is too high at level 0, its size can be reduced

            # if mem energy consumption is too high at level 1, bandwidth/frequency can be reduced

            ## What is really high read energy, write energy or leakage energy -> which depends on the leakage time
            # If leakage energy, read or write energy-> can change the technology type

        if opts == "time":
            print(
                scheduler.bandwidth_idle_time,
                scheduler.compute_idle_time,
                scheduler.mem_size_idle_time,
            )

        ## If compute idle time, Update Compute

        ## If Bandwith Limited update memory banks, only if memory banks are off chip then connectivity is forced
        ## Frequency can be increased, Have a number of smaller arrays, increases energy consumption ->
        ## find the technology space that reduces frequency

        for i in range(self.mle):
            memory = config["memory"]["level" + str(i)]
            self.mem_read_bw[i] = (
                memory["frequency"]
                * memory["banks"]
                * memory["read_ports"]
                * memory["width"]
            )
            self.mem_write_bw[i] = (
                memory["frequency"]
                * memory["banks"]
                * memory["write_ports"]
                * memory["width"]
            )

        ## If Mem Size idle time, Update mem size, update of size is proportional to the sizing of the memory

        newhw = scheduler.config

        return newhw

    def findtechnology(self, opts):

        """
        opts is of either frequency or its for energy -> can modulate the access time of the cell and the cell energy
        The technology space can be loaded from the file, and how we can rapidly find our point there
        The wire space is also loaded, and the joint technology and wire space can also be loaded
        """
        ## For joint optimization of technology space and wire space, we produce the sensitivity analysis,
        # of technology space and wire space
        ## and how it affects the design space in energy of access
        ## It can affect the frequency in the sizing of the memory arrays, it affect the energy of accesses also

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

    def get_mem_energy(self, size, read_ports, banks):
        return mem_table[i]

    # def get_compute_energy(self, *args, **kwargs):
    #     pass

    # def get_compute_area(self, *args, **kwargs):
    #     pass
    # def get_mem_area(self, *args, **kwargs):
    #     pass
