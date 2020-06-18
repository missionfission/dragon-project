import numpy as np
import yaml
import yamlordereddictloader


class Generator:
    def __init__(self, constraintfiles=["max.yaml", "min.yaml"]):
        base_dir = "configs/"

        self.maxval = yaml.load(
            open(base_dir + constraintfiles[0]), Loader=yamlordereddictloader.Loader
        )
        self.minval = yaml.load(
            open(base_dir + constraintfiles[1]), Loader=yamlordereddictloader.Loader
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
        Time lost due to small memory size ?
        Time lost due to small memory bandwidth ?
        Time lost due to high of everything but compute is slow
        Energy high due to small memory size ?
        Energy high due to small memory bandwidth ?
        Energy high due to high of everything but compute is slow
        Where is area getting consumed the most?
        """
        compute_idle_time()
        increase_memory_size
        increase_memory_bandwidth
        increase_compute_arrays
        # Check if noc time is ever going to be a problem ?
        # Check whether minval and maxval satisfy here ?
        arevaluesrealistic
        return newhw

    def save_statistics(self, scheduler):
        """
        Execution statistics also have to be generated : Area, Energy, Time/Number of Cycles 
        """
        mem_area = np.zeros((scheduler.mle))
        compute_area = 0
        total_area = np.sum(mem_area) + compute_area

        mem_energy = np.zeros((scheduler.mle))
        compute_energy = 0
        total_energy = np.sum(mem_energy) + compute_energy

        scheduler.logger.info("Tool Output")
        scheduler.logger.info("===========================")
        scheduler.logger.info("Total No of cycles  = %d ", scheduler.total_cycles)
        scheduler.logger.info("Memory Energy Consumption  = %d ", np.sum(mem_energy))
        scheduler.logger.info("Compute Energy Consumption  = %d ", compute_energy)
        scheduler.logger.info(" Total Energy Consumption  = %d ", total_energy)
        scheduler.logger.info("Memory Area Consumption  = %d ", np.sum(mem_area))
        scheduler.logger.info("Compute Area Consumption  = %d ", compute_area)
        scheduler.logger.info("Total Area Consumption  = %d ", total_area)
