import glob
import os
import sys
from collections import deque
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import yaml
import yamlordereddictloader
from torchvision import models
from yaml import dump

from ir.handlers import handlers
from ir.trace import trace
from scheduling import Scheduling


def set_node_characterstics(graph):
    for node in graph.nodes:
        for operators, func in handlers:
            if isinstance(operators, str):
                operators = [operators]
            if node.operator in operators:
                if func is not None:
                    print(node.operator)
                    node.compute_expense, node.read_access, node.write_access = func(
                        node
                    )


def runner():
    """
    Runs the Input Graph
    """
    #     logger = Logger()

    #     hwdesc = yaml_parser(filename)
    add_plug_ins = [
        "accelergy_ART",
        "accelergy_ERT",
        "cacti_memory",
        "orion_noc",
        "aladdin_compute",
    ]
    #     plugins.instatiate_plugins(add_plug_ins)
    #     if hwdesc != None:
    #         print("Mapping the Model on the Given Hardware")
    #     else:
    #         print("Generating the Hardware Description and Logging Statistics")

    # For the Inverse Problem
    maxconstraints = yaml.load(open("max.yaml"), Loader=yamlordereddictloader.Loader)
    #     minconstraints = yaml.load(open('min.yaml'), Loader=yamlordereddictloader.Loader)
    #   Create Graph

    for name, model in models.__dict__.items():
        if not name.islower() or name.startswith("__") or not callable(model):
            continue
        model = model().eval()
        if "resnet50" in name:
            inputs = torch.randn(1, 3, 299, 299)
            graph = trace(model, inputs)
    #   Set Node Characterstics
    set_node_characterstics(graph)
    scheduler = Scheduling()
    scheduler.run(graph)
    # For the Forward Problem
    # executer = Mapper(opts, hwdesc)
    # logger.save_statistics(area)
    # logger.save_statistics(energy)
    # logger.save_statistics(timing)


runner()
