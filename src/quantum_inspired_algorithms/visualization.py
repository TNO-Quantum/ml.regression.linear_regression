from pathlib import Path
from typing import Optional
import matplotlib
import pandas as pd
import numpy as np
import seaborn as sns

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from numpy.typing import NDArray

plt.rcParams["font.size"] = "4"


def _get_plot_dir():
    plots_dir = Path("plots")
    plots_dir.mkdir(exist_ok=True)
    return plots_dir


def _save_fig(name: str):
    plots_dir = _get_plot_dir()
    plt.savefig(Path(plots_dir, name), dpi=600, bbox_inches="tight")
    plt.close("all")


def _scatter_plot(data: pd.DataFrame):
    sns.scatterplot(data=data, x="expected", y="actual")
    min_value = np.min([np.min(data["actual"]), np.min(data["expected"])])
    max_value = np.max([np.max(data["actual"]), np.max(data["expected"])])
    plt.plot(
        [min_value, max_value],
        [min_value, max_value],
        "r-",
    )


def plot_solution(
    x_reference: NDArray,
    best_idx: NDArray,
    plot_name: str,
    expected_solution: Optional[NDArray] = None,
    solution: Optional[NDArray] = None,
    expected_counts: Optional[NDArray] = None,
    counts: Optional[NDArray] = None,
) -> int:
    """Plot (sorted) reference solution along with best indices provided."""
    # Squeeze array
    x_reference = np.squeeze(np.abs(x_reference))

    # Define colors for plotting
    colors = np.array(x_reference.size * ["blue"])
    colors[best_idx] = "red"

    # Find mapping from index in sorted array to original index
    sort_idx = np.flip(np.argsort(x_reference))

    # Sort in descending order
    sorted_x_known = x_reference[sort_idx]
    sorted_colors = colors[sort_idx]

    # Plot (sorted) reference solution and best indices provided
    sns.lineplot(sorted_x_known)
    for idx, value in enumerate(sorted_colors):
        if value == "red":
            plt.axvline(x=idx, color=value, linestyle="-", linewidth=0.2)
    n_best_idx = len(best_idx)
    n_matches = np.sum(sorted_colors[:n_best_idx] == "red")
    plt.title(f"{n_matches} out of {n_best_idx}")
    _save_fig(f"{plot_name}_matches")

    if solution is not None and expected_solution is not None:
        # Plot solution
        solution_df = pd.DataFrame({"actual": solution, "expected": expected_solution})
        _scatter_plot(solution_df)
        _save_fig(f"{plot_name}_solution_scatter")

    if counts is not None and expected_counts is not None:
        # Plot counts
        counts_df = pd.DataFrame({"actual": counts, "expected": expected_counts})
        _scatter_plot(counts_df)
        _save_fig(f"{plot_name}_counts_scatter")

    return int(n_matches)
