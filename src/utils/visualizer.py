# Generate a Time Lapse Video
from collections import namedtuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.ticker import FormatStrFormatter, LinearLocator, MaxNLocator
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import


def roofline_model():
    pass


def generate_execution_movie():
    pass


def genvideologger():
    pass


def bandwidth_bar_graph(
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


def cycles_bar_graph(
    filename,
    cycles,
    free_cycles,
    xticklabels=None,
    xlabel=None,
    ylabel=None,
    bar_width=0.2,
):
    fig, ax = plt.subplots(figsize=(30, 10))
    base_dir = "figures/"
    error_config = {"ecolor": "0.3"}
    index = np.arange((len(cycles)))
    ax.bar(
        index,
        cycles,
        bar_width,
        color="blue",
        error_kw=error_config,
        label="Actual Cycles Required for Execution",
    )
    ax.bar(
        index + 2 * bar_width,
        free_cycles,
        bar_width,
        color="red",
        error_kw=error_config,
        label="Free Cycles for Execution",
    )
    ax.legend(fontsize=20)
    # plt.yscale("log")
    ax.set_xticks(index)
    ax.set_xticklabels(xticklabels)
    plt.xticks(rotation=80)
    plt.rc("xtick", labelsize=20)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=20)
    ax.set_xlabel("Graph Nodes", fontsize=20, fontweight="bold")
    fig.tight_layout()
    plt.savefig(base_dir + filename, bbox_inches="tight")
    plt.show()


def mem_util_bar_graph(
    filename,
    cycles,
    free_cycles,
    xticklabels=None,
    xlabel=None,
    ylabel=None,
    bar_width=0.2,
):
    fig, ax = plt.subplots(figsize=(30, 10))
    base_dir = "figures/"
    error_config = {"ecolor": "0.3"}
    index = np.arange((len(cycles)))
    ax.bar(
        index,
        cycles,
        bar_width,
        color="blue",
        error_kw=error_config,
        label="Actual Memory Utilization in Execution",
    )
    ax.bar(
        index + 2 * bar_width,
        free_cycles,
        bar_width,
        color="red",
        error_kw=error_config,
        label="Memory Utilization without Prefetching",
    )
    ax.legend(fontsize=20)
    plt.yscale("log")
    ax.set_xticks(index)
    ax.set_xticklabels(xticklabels)
    plt.xticks(rotation=80)
    plt.rc("xtick", labelsize=20)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=20)
    ax.set_xlabel("Graph Nodes", fontsize=20, fontweight="bold")
    fig.tight_layout()
    plt.savefig(base_dir + filename, bbox_inches="tight")
    plt.show()


def tech_space_graph(
    filename,
    wire_space,
    memory_cell_space,
    cmos_space,
    xticklabels=None,
    xlabel=None,
    ylabel=None,
    bar_width=0.2,
):
    fig = plt.figure()
    ax = fig.gca(projection="3d")

    # Plot the surface.
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax.set_zlim(-1.01, 1.01)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter("%.02f"))

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()

    # ax.legend(fontsize=20)
    # plt.yscale("log")
    # ax.set_xticks(index)
    # ax.set_xticklabels(xticklabels)
    # plt.xticks(rotation=80)
    # plt.rc("xtick", labelsize=20)  # fontsize of the tick labels
    # plt.rc("ytick", labelsize=20)
    # ax.set_xlabel("Graph Nodes", fontsize=20, fontweight="bold")
    # fig.tight_layout()
    # plt.savefig(base_dir + filename, bbox_inches="tight")


def plot_descent(
    time_list, bandwidth_time_list, mem_size_idle_time_list, *args, **kwargs
):
    fig, ax = plt.subplots(figsize=(30, 10))
    base_dir = "figures/"
    error_config = {"ecolor": "0.3"}
    index = np.arange((len(time_list)))
    plt.plot(time_list)
    plt.plot(bandwidth_time_list)
    plt.plot(mem_size_idle_time_list)
    ax.legend(fontsize=20)
    ax.set_xticks(index)
    plt.xticks(rotation=80)
    plt.rc("xtick", labelsize=20)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=20)
    ax.set_xlabel("Grad descent time", fontsize=20, fontweight="bold")
    fig.tight_layout()
    plt.savefig(base_dir + "time.png", bbox_inches="tight")
    plt.show()


def plot_parameter_change(bank_list, mem_size_list, compute_list, *args, **kwargs):
    fig, ax = plt.subplots(figsize=(30, 10))
    base_dir = "figures/"
    error_config = {"ecolor": "0.3"}
    index = np.arange((len(bank_list)))
    plt.plot(bank_list)
    # plt.plot(mem_size_list)
    # plt.plot(mem_size_idle_time_list)
    ax.legend(fontsize=20)
    ax.set_xticks(index)
    plt.xticks(rotation=80)
    plt.rc("xtick", labelsize=20)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=20)
    ax.set_xlabel("Grad descent time", fontsize=20, fontweight="bold")
    fig.tight_layout()
    plt.savefig(base_dir + "banks.png", bbox_inches="tight")
    plt.show()


def plot_parameter_change_multiple(
    ax, bank_list, mem_size_list, compute_list, *args, **kwargs
):
    ax.plot(bank_list, "go-")
    # ax.plot(mem_size_list[1:], "go-")


def plot_descent_multiple(
    ax, time_list, bandwidth_time_list, mem_size_idle_time_list, *args, **kwargs
):
    ax.plot(time_list[1:], "ro-")
    ax.plot(bandwidth_time_list, "bo-")
    # ax.plot(mem_size_idle_time_list[1:], "bo-")
