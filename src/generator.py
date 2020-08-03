import numpy as np
import pandas as pd
import yaml
import yamlordereddictloader

mem_table = np.array(pd.read_csv("tables/sram.csv"))


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
        opts = ["energy", "time", "area", "edp"]
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
        newhw = scheduler.config
        mm_compute = config["mm_compute"]
        vector_compute = config["vector_compute"]

        if config["mm_compute"]["class"] == "systolic_array":
            config["mm_compute_per_cycle"] = (
                ((mm_compute["size"]) ** 2) * mm_compute["N_PE"] / 2
            )
            config["comp_bw"] = (
                mm_compute["size"] * mm_compute["N_PE"] * mm_compute["frequency"] * 2
            )

            self.logger.info(
                "MM Compute per cycle : %d", config["mm_compute_per_cycle"]
            )
            self.logger.info("Compute Bandwidth Required : %d", config["comp_bw"])

        if config["mm_compute"]["class"] == "mac":
            config["mm_compute_per_cycle"] = (
                ((mm_compute["size"]) ** 2) * mm_compute["N_PE"] / 2
            )
            config["comp_read_bw"] = (
                mm_compute["size"] * mm_compute["N_PE"] * mm_compute["frequency"] * 2
            )

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
            self.mem_size[i] = memory["size"]

        return newhw

    def findtechnologyparameters(self):
        """

        have to really think, what is the technology space, and how we can rapidly find our point there 
        """
        pass

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
        compute_energy = self.get_compute_energy(
            mm_compute["class"], mm_compute["size"], mm_compute["N_PE"]
        )

        for i in range(scheduler.mle):
            memory = config["memory"]["level" + str(i)]
            mem_energy_access[i] = self.get_mem_energy(
                memory["size"], memory["read_ports"], memory["banks"], connectivity,
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

    # def get_mem_energy(self, *args, **kwargs):
    #     pass
    # def get_compute_energy(self, *args, **kwargs):
    #     pass

    # def get_compute_area(self, *args, **kwargs):
    #     pass
    # def get_mem_area(self, *args, **kwargs):
    #     pass

