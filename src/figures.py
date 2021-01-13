import os
from collections import namedtuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

path = os.path.join(os.path.dirname(__file__))
print(path)
fig, ax = plt.subplots()
ax.plot(
    [1, 2.3, 4.18, 4.18], [1, 9.5, 9.5, 26.1], "ro-", label="Derived Technology Targets"
)
ax.plot(
    [1, 1, 1.81], [1, 1.01, 1.01], "bo-", label="Other Technology Improvement Paths"
)
ax.plot([1, 1.2, 1.2, 3.4], [1, 3, 7, 7], "bo-")
ax.plot([1, 2, 2, 3.5], [1, 3, 6, 12], "bo-")
ax.set_ylim(1, 30)
ax.set_xlim(1, 5)
ax.set_ylabel("Energy Efficiency", fontsize=16, fontweight="bold")
ax.set_xlabel("Execution Time", fontsize=16, fontweight="bold")
plt.rc("xtick", labelsize=16)  # fontsize of the tick labels
plt.rc("ytick", labelsize=16)
ax.legend(fontsize=12)
plt.savefig("./figures/paths.png", bbox_inches="tight")
plt.show()
