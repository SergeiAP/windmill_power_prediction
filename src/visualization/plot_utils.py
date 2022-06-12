# pylint: disable=missing-module-docstring
from pathlib import Path

import matplotlib.pyplot as plt
from plotly.graph_objs._figure import Figure as plotly_Figure
import matplotlib.axes as mpl_axes
import matplotlib.figure as mpl_figure
import seaborn as sns
from pylab import rcParams  # type: ignore


def set_plot_params(figsize: tuple[int, int] = (20, 10)) -> None:
    """Set default params for better plot representation

    Args:
        figsize (tuple[int, int]): default size of plots
    """

    small_size = 16
    medium_size = 18
    bigger_size = 20

    rcParams['font.size'] = small_size          # controls default text sizes
    rcParams['axes.titlesize'] = small_size     # fontsize of the axes title
    rcParams['axes.labelsize'] = medium_size    # fontsize of the x and y labels
    rcParams['xtick.labelsize'] = small_size    # fontsize of the tick labels
    rcParams['ytick.labelsize'] = small_size    # fontsize of the tick labels
    rcParams['legend.fontsize'] = small_size    # legend fontsize
    rcParams['figure.titlesize'] = bigger_size  # fontsize of the figure title
    
    rcParams['figure.figsize'] = figsize
    plt.set_loglevel('WARNING') # type: ignore
    sns.set_theme()


def save_plot(
    figure: mpl_figure.Figure | sns.axisgrid.PairGrid | mpl_axes.Axes | plotly_Figure,
    figname: str | Path) -> None:
    """Save figure

    Args:
        figure (mpl_figure.Figure | sns.axisgrid.PairGrid | mpl_axes.Axes): what to 
        save
        figname (str | Path): path to save
    """
    match figure:
        case sns.axisgrid.PairGrid():
            figure.savefig(figname)
        case mpl_figure.Figure() | mpl_axes.Axes():
            figure.get_figure().savefig(figname)
        case plotly_Figure():
            figure.write_html(figname)
    print(f"Save plot as {figname}")
