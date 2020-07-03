# Generate a Time Lapse Video
from collections import namedtuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator


def roofline_model():

    pass


def generate_execution_movie():
    pass


def genvideologger():

    pass


def bar_graph(filename, var, xticklabels=None, xlabel=None, ylabel=None, bar_width=0.2):
    fig, ax = plt.subplots(figsize=(30, 10))
    base_dir = "figures/"
    error_config = {"ecolor": "0.3"}
    index = np.arange((len(var)))
    ax.bar(
        index, var, bar_width, color="blue", error_kw=error_config, label="",
    )
    ax.set_xticks(index)
    ax.set_xticklabels(xticklabels)
    plt.xticks(rotation=80)
    plt.rc("xtick", labelsize=20)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=20)
    ax.set_ylabel("Read Bandwidth", fontsize=20)
    ax.set_xlabel("Graph Nodes", fontsize=20)
    fig.tight_layout()
    plt.savefig(base_dir + filename, bbox_inches="tight")
    plt.show()
