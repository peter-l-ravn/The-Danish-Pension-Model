import numpy as np
import matplotlib.pyplot as plt

def plot_extensive_intensive(share_working, hours_if_work):
    """
    Plot extensive margin (share working) as a line and
    intensive margin (hours if working) as a bar plot.
    """

    share_working = np.asarray(share_working)
    hours_if_work = np.asarray(hours_if_work)

    if share_working.shape != hours_if_work.shape:
        raise ValueError("share_working and hours_if_work must have the same shape.")

    x = np.arange(len(share_working))

    fig, ax = plt.subplots(figsize=(8, 5))

    # Bar plot for intensive margin
    ax.plot(
        x,
        hours_if_work,
        color="steelblue",
        linewidth=2.5,
        label="Intensive margin (hours if working)"
    )

    # Line plot for extensive margin (light turquoise)
    ax.plot(
        x,
        share_working,
        color="indianred",
        linewidth=2.5,
        linestyle='dashed',
        # marker="-",
        label="Extensive margin (share working)"
    )

    ax.set_ylim(0.0, 1.2)

    # Ensure exactly 1-unit tick spacing
    ax.set_xticks(x)

    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    ax.set_title("Extensive and Intensive Labor Supply Margins")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(frameon=False)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.show()


import numpy as np
import matplotlib.pyplot as plt

def plot_value_functions(model, i):
    """
    Plot value functions for working and not working
    for a given index i over the asset grid.
    """

    a_grid = np.asarray(model.par.a_grid)
    V_work = np.asarray(model.sol.V_work[i])
    V_notwork = np.asarray(model.sol.V_notwork[i])

    if not (a_grid.shape == V_work.shape == V_notwork.shape):
        raise ValueError("Asset grid and value functions must have the same shape.")

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(
        a_grid,
        V_work,
        color="steelblue",
        linewidth=2.5,
        label="Value (working)"
    )

    ax.plot(
        a_grid,
        V_notwork,
        color="indianred",
        linewidth=2.5,
        linestyle="dashed",
        label="Value (not working)"
    )

    ax.set_xlabel("Assets")
    ax.set_ylabel("Value")
    ax.set_title(f"Value Functions, index = {i}")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(frameon=False)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.show()
