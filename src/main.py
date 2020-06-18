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

from generator import Generator
from ir.handlers import handlers, set_node_characterstics
from ir.trace import trace
from scheduling import Scheduling


def runner(graph_set):
    """
    Runs the Input Graph
    """

    """
    Scheduling works in the following way :
    1. Start with a given/random Hardware point -> Nodes of the graph are scheduled (prefetching)
    
    2. Do the Scheduling with that Point -> Mapping stops here -> Further evaluation is done using accelergy 
    (with values taken from ERT/ART) -> If values not available -> Use plugins for generating these values 
    
    3. Log bottlenecks and work on a different Hardware point -> do this till some realistically
    max, min values are not violated -> Values/Analyses for a different/unavailable point will require full 
    From the technology node set, generate the reference tables using plugins integration of plugins -> 
    Currently using a table at 40nm.   
    
    4. Optimization metric (time/area/energy) of execution in various components, 
    then optimize the metric of execution and take decisions accordingly 
    
    5. Optimize over different workloads ?
    
    """

    num_iterations = 5
    generator = Generator()
    for graph in graph_set:
        scheduler = Scheduling()
        scheduler.run(graph)
        generator.save_statistics(scheduler)
        for i in range(num_iterations):
            nexthw = generator.findnext(scheduler)
            generator.writehwfile(nexthw, "iter" + str(i) + "hw.yaml")
            scheduler.create_config(nexthw)
            scheduler.run(graph)
            generator.save_statistics(scheduler)


####################################

for name, model in models.__dict__.items():
    if not name.islower() or name.startswith("__") or not callable(model):
        continue
    model = model().eval()
    if "resnet50" in name:
        inputs = torch.randn(1, 3, 100, 100)
        graph = trace(model, inputs)

runner([graph])
