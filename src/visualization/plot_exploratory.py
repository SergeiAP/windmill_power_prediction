# pylint: disable=missing-module-docstring
# TODO: change print to log
import time
from pathlib import Path

import click
import matplotlib.axes as mpl_axes
import matplotlib.figure as mpl_figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pylab import rcParams  # type: ignore

from plot_utils import save_plot, set_plot_params # pylint: disable=import-error
from src.read_config import get_data_config


def plot_corr_matrix(df: pd.DataFrame,
                     title: str = '',
                     ax: np.ndarray | None = None,
                     figsize: tuple = (30, 15),
                     fontsize: int = 14
                     ) -> mpl_axes.Axes: # pylint: disable=no-member
    """Plot correlation matrix

    Args:
        df (pd.DataFrame): dataset with columns to be displayed
        title (str, optional): title for f'Correlation Heatmap {title}' form.
        Defaults to ''.
        ax (np.ndarray | None, optional): matplotlib axis. Defaults to None.
        figsize (tuple, optional): size of the plot. Defaults to (20, 10).

    Returns:
        mpl_axes.Axes: matplotlib axis
    """
    plt.figure(figsize=figsize)
    # To show one half only
    mask = np.triu(np.ones_like(df.corr(), dtype=bool))
    heatmap: mpl_axes.Axes = sns.heatmap(round(df.corr(), 2), # type: ignore
                                         mask=mask,
                                         annot_kws={"fontsize": fontsize},
                                         annot=True, ax=ax)
    # Give a title to the heatmap.
    # Pad defines the distance of the title from the top of the heatmap.
    heatmap.set_title(f'Correlation Heatmap {title}', fontdict={'fontsize': 16}, pad=12)
    return heatmap


def plot_categorical_corr_matrix(df: pd.DataFrame,
                                 cat_col: str,
                                 figsize: tuple = (30, 10),
                                 fontsize: int = 14
                                 ) -> mpl_figure.Figure: # pylint: disable=no-member
    """Plot correlation matrix for each group in `cat_col`

    Args:
        df (pd.DataFrame): data with `cat_col`
        cat_col (str): column with categorical feature to create it's own corr_matrix
        figsize (tuple, optional): one subplot size. Defaults to (30, 10).
        fontsize (int, optional): fontsize for corr value. Defaults to 14.

    Returns:
        mpl_figure.Figure: correlation matrixes to save
    """
    categories = df[cat_col].unique()
    figsize_all = (figsize[0], figsize[1]*len(categories))
    fig, ax  = plt.subplots(len(categories), 1, figsize=figsize_all)
    fig.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9,
                        wspace=0.2, hspace=0.4)
    for idx, category in enumerate(categories):
        df_to_show = df.loc[df[cat_col] == category]
        plot_corr_matrix(df=df_to_show, 
                         title=f"for {cat_col} {category}",
                         ax=ax[idx], # type: ignore
                         figsize=figsize,
                         fontsize=fontsize)
    return fig


def plot_boxplot(df: pd.DataFrame,
                 is_standardized: bool = True,
                 figsize=(20, 10)) -> mpl_figure.Figure:
    """Plot boxplots for presented in dataframe features

    Args:
        df (pd.DataFrame): dataframe to be represented in boxplot
        is_standardized (bool, optional): standardize vals by `(val - mean)/std`.
        Defaults to True.
        figsize (tuple, optional): Size of the picture. Defaults to (20, 10).

    Returns:
        mpl_figure.Figure: matplotlib figure to save
    """
    if is_standardized:
        df = df.select_dtypes(include=np.number) # type: ignore
        df = (df - df.mean(axis="index")) / df.std(axis="index")
    fig, ax = plt.subplots(figsize=figsize)
    fig.subplots_adjust(bottom=0.2)
    ax.tick_params(axis='x', rotation=45)
    ax = sns.boxplot(data=df, notch=True, ax=ax)
    return fig


def plot_pairplot(df: pd.DataFrame,
                  hue_col: str,
                  kind: str = 'reg',
                  diag_kind: str = 'kde',
                  alpha: float = 0.3) -> sns.axisgrid.PairGrid:
    """Plot pairwise relationships in a dataset in triangular format 

    Args:
        df (pd.DataFrame): dataset with columns to be ploted
        hue_col (str): columns denoted classes and show them by diffrent colors
        kind (str, optional): kind of plots under the diagonal. Defaults to 'reg'.
        diag_kind (str, optional): kind of plots on the diagonal. Defaults to 'kde'.
        alpha (float, optional): transparency of elements. Defaults to 0.3.

    Returns:
        sns.axisgrid.PairGrid: seaborn pairplot to save
    """
    pairplot = sns.pairplot(df,
                            hue=hue_col, 
                            kind=kind,
                            diag_kind=diag_kind, 
                            plot_kws={'scatter_kws': {'alpha': alpha}}, 
                            corner=True)
    sns.move_legend(pairplot, "upper right", bbox_to_anchor=(0.95, 0.95))
    return pairplot


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("output_folder", type=click.Path())
def run_exploratory_plots(input_filepath: str, output_folder: str) -> None:
    """Run exploratory analysis in the dataset `input_filepath`

    Args:
        input_filepath (str): path to dataset with features
        output_folder (str): folder to save pictures
    """
    start_time = time.time()
        
    # read section
    save_folder = Path(Path(".") / output_folder)
    df = pd.read_csv(input_filepath)
    (features,
     set_plot_params_cfg,
     plot_corr_matrix_cfg,
     plot_boxplot_cfg,
     plot_pairplot_cfg) = get_data_config("plot_exploratory", 
         ["features",
          "set_plot_params",
          "plot_corr_matrix",
          "plot_boxplot",
          "plot_pairplot"]
    )
    features: dict = features
    (target, ) = get_data_config("common", ["target"])
     
    # target is last column
    df = (df.loc[:, features["include"] + [target]] 
          if isinstance(features["include"], list) else 
          df.loc[:, df.columns.drop(target).to_list() + [target]])
    
    # TODO: make several if's using dict and list of required plots
    if set_plot_params_cfg["is_on"]:
        set_plot_params()
    if plot_corr_matrix_cfg["is_on"]:
        heatmap_plot = plot_corr_matrix(df,
                                        "for all features", 
                                        figsize=plot_corr_matrix_cfg["figsize"],
                                        fontsize=plot_corr_matrix_cfg["fontsize"])
        save_plot(heatmap_plot, save_folder / "exp_all_feature_corr_matrix.png")
        if plot_corr_matrix_cfg["cat_col"]:
            heatmap_plot = plot_categorical_corr_matrix(
                df,
                plot_corr_matrix_cfg["cat_col"],
                figsize=plot_corr_matrix_cfg["figsize"],
                fontsize=plot_corr_matrix_cfg["fontsize"]
            )
            save_plot(heatmap_plot, save_folder / "exp_categorical_corr_matrix.png")
    if plot_boxplot_cfg["is_on"]:
        boxplot = plot_boxplot(df, plot_boxplot_cfg["is_standardized"])
        save_plot(boxplot, save_folder / "exp_boxplot.png")
    if plot_pairplot_cfg["is_on"]:
        del plot_pairplot_cfg["is_on"]
        pairplot = plot_pairplot(df=df, **plot_pairplot_cfg)
        save_plot(pairplot, save_folder / "exp_pairplot.png")

    print(f"Execution time is {round((time.time() - start_time) / 60, 2)} minutes")

if __name__ == "__main__":
    run_exploratory_plots()  # pylint: disable=no-value-for-parameter
    