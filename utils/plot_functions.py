import matplotlib.pyplot as plt
import numpy as np

from utils import logging, plot_tools

logger = logging.child_logger(__name__)


def plot_matrix(matrix, labels=[], matrix_type="correlation", name="", outDir="./"):
    logger.info(f"Make {matrix_type} matrix plot")

    fig, ax = plt.subplots()
    ax.imshow(matrix)

    # Show all ticks and label them with the respective list entries
    ax.set_xticks(np.arange(len(labels)), labels=labels)
    ax.set_yticks(np.arange(len(labels)), labels=labels)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    if matrix_type == "correlation":
        nround = lambda x: round(x, 3)
    else:
        nround = lambda x: round(x, 2)

    # Loop over data dimensions and create text annotations.
    for i in range(len(labels)):
        for j in range(len(labels)):

            val = nround(matrix[i, j])
            ax.text(j, i, val, ha="center", va="center", color="w")

    # ax.set_title("Harvest of local farmers (in tons/year)")
    fig.tight_layout()
    name = name if name.startswith("_") else "_" + name

    plot_tools.save_plot(outDir, f"{matrix_type}{name}")
