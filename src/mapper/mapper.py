import collections
import logging
import pdb

import numpy as np
import yaml
import yamlordereddictloader

from generator import *
from generator import Generator, get_mem_props
from synthesis import ai_utils
from utils.logger import create_logger
from utils.visualizer import *

eff = 0.5


class Mapper:
    def __init__(self, hwfile="default.yaml", stats_file="logs/stats.txt"):
        """Mapper Class that Provides Interface for Different Mappings on the Architecture 

        Args:
            hwfile (str, optional): [description]. Defaults to "default.yaml".
            stats_file (str, optional): [description]. Defaults to "logs/stats.txt".
        """
        base_dir = "configs/"
        self.total_cycles = 0
        self.technology = [1, 1, 40]
        # maybe change this later to peripheral logic node or speed
        #     [wire_cap , sense_amp_time, plogic_node],
        self.logger = create_logger(stats_file=stats_file)
        self.config = self.complete_config(
            yaml.load(open(base_dir + hwfile), Loader=yamlordereddictloader.Loader)
        )

    def complete_config(self, config):
        """
        1. Completes the Config for Hardware Description by using Technology/Design Functions
        2. Adds State Variables as Args that are Tracked for ASAP Mapping

        Args:
            config ([type]): [description]

        Returns:
            [type]: [description]
        """

        self.logger.debug("Config Statistics : ")

        self.mle = config["memory_levels"]
        self.mem_energy = np.zeros((self.mle))
        self.compute_energy = 0
        self.mem_read_access = np.zeros((self.mle))
        self.mem_write_access = np.zeros((self.mle))
        self.mem_size = np.zeros((self.mle))
        self.mem_util = np.zeros((self.mle))
        self.mem_free = np.zeros((self.mle))
        self.mem_read_bw = np.zeros((self.mle))
        self.mem_write_bw = np.zeros((self.mle))
        self.internal_bandwidth_time = 0
        self.total_cycles = 0
        self.bandwidth_idle_time = 0
        self.compute_idle_time = 0
        self.mem_size_idle_time = 0

        self.force_connectivity = config["force_connectivity"]
        mm_compute = config["mm_compute"]
        vector_compute = config["vector_compute"]

        if mm_compute["class"] == "systolic_array":
            config["mm_compute_per_cycle"] = (
                ((mm_compute["size"]) ** 2) * mm_compute["N_PE"] / (4)
            )
            config["comp_bw"] = (
                mm_compute["size"]
                * mm_compute["N_PE"]
                * mm_compute["frequency"]
                * 2
                / 4
            )

            self.logger.debug(
                "MM Compute per cycle : %d", config["mm_compute_per_cycle"]
            )
            self.logger.debug("Compute Bandwidth Required : %d", config["comp_bw"])

        if config["mm_compute"]["class"] == "mac":
            config["mm_compute_per_cycle"] = (mm_compute["size"]) * mm_compute["N_PE"]
            config["comp_read_bw"] = (
                mm_compute["size"] * mm_compute["N_PE"] * mm_compute["frequency"]
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

            self.logger.debug(
                "Memory at Level %d, Read Bandwidth %d Write Bandwidth %d",
                i,
                self.mem_read_bw[i],
                self.mem_write_bw[i],
            )
        # complete_functional_config
        # complete_performance_config
        # memory
        for i in range(self.mle - 1):
            memory = config["memory"]["level" + str(i)]
            read_energy, write_energy, leakage_power, area = get_mem_props(
                memory["size"], memory["width"], memory["banks"]
            )
            config["memory"]["level" + str(i)]["read_energy"] = str(read_energy)
            config["memory"]["level" + str(i)]["write_energy"] = str(write_energy)
            config["memory"]["level" + str(i)]["leakage_power"] = str(leakage_power)
            config["memory"]["level" + str(i)]["area"] = str(area)
        # compute
        # config["memory"] = mem_space(config["memory"], technology)
        # config["mm_compute"] = comp_space(config["mm_compute"], technology)
        return config
