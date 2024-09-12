from pathlib import Path
from typing import Optional
import matplotlib
import numpy as np
import pandas as pd
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


def compute_n_matches(x_reference: NDArray, best_idx: NDArray, plot_name: Optional[str] = None) -> int:
    """Compute number of matches."""
    # Compute absolute value
    x_reference = np.squeeze(np.abs(x_reference))

    # Mark indices assumed to be best
    marked_as_best = np.zeros(x_reference.size, dtype=bool)
    marked_as_best[best_idx] = True

    # Find mapping from index in sorted array to original index
    sort_idx = np.flip(np.argsort(x_reference))

    # Sort in descending order
    sorted_marked_as_best = marked_as_best[sort_idx]
    n_best_idx = len(best_idx)

    # Compute number of matches
    n_matches = int(np.sum(sorted_marked_as_best[:n_best_idx]))

    # Plot (sorted) reference solution and best indices provided
    if plot_name is not None:
        sorted_x_known = x_reference[sort_idx]
        sns.lineplot(sorted_x_known)
        for idx, value in enumerate(sorted_marked_as_best):
            if value:
                plt.axvline(x=idx, color="red", linestyle="-", linewidth=0.2)
        plt.title(f"{n_matches} out of {n_best_idx}")
        _save_fig(f"{plot_name}_matches")

    return n_matches


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
    # Compute number of matches
    n_matches = compute_n_matches(x_reference, best_idx, plot_name=plot_name)

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

    return n_matches
