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


def bar_graph(
    filename,
    var1,
    var2,
    var3,
    xticklabels=None,
    xlabel=None,
    ylabel=None,
    bar_width=0.2,
):
    fig, ax = plt.subplots(figsize=(30, 10))
    base_dir = "figures/"
    error_config = {"ecolor": "0.3"}
    index = np.arange((len(var1)))
    plt.axhline(y=var3, color="black", linestyle="-", label="Limit", linewidth=6)
    ax.bar(
        index,
        var1,
        bar_width,
        color="blue",
        error_kw=error_config,
        label="Bandwidth Requirements",
    )
    ax.bar(
        index + 2 * bar_width,
        var2,
        bar_width,
        color="red",
        error_kw=error_config,
        label="Actual Bandwidth in Execution",
    )
    ax.legend(fontsize=20)
    plt.yscale("log")
    ax.set_xticks(index)
    ax.set_xticklabels(xticklabels)
    plt.xticks(rotation=80)
    plt.rc("xtick", labelsize=20)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=20)
    ax.set_ylabel("Read Bandwidth", fontsize=20, fontweight="bold")
    ax.set_xlabel("Graph Nodes", fontsize=20, fontweight="bold")
    fig.tight_layout()
    plt.savefig(base_dir + filename, bbox_inches="tight")
    plt.show()
