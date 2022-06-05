# TODO: covert print to log
import time
from pathlib import Path

import click
import matplotlib.axes as mpl_axes
import matplotlib.figure as mpl_figure
import matplotlib.pyplot as plt
import pandas as pd
import ppscore as pps
import seaborn as sns
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from sklearn.utils._bunch import Bunch as sklearn_bunch
from src.read_config import get_data_config

from plot_exploratory import save_plot, set_plot_params


def get_permeation_importance(df: pd.DataFrame,
                              target_col: str,
                              train_test_cfg: dict,
                              random_forest_cfg: dict,
                              n_repeats: int = 10,
                              random_state: int = 42) -> sklearn_bunch:
    feature_cols = df.columns.drop(target_col)
    X_train, X_test, y_train, y_test = train_test_split(df[feature_cols],
                                                        df[target_col],
                                                        **train_test_cfg)
    rf = ExtraTreesRegressor(random_state=random_state,
                             **random_forest_cfg)
    rf.fit(X_train, y_train)
    print(f"Permeation: random forest score is "
          f"{mean_absolute_percentage_error(rf.predict(X_test), y_test)}")
    
    result = permutation_importance(
        rf, X_test, y_test, n_repeats=n_repeats, random_state=random_state, n_jobs=-1)
    return result


def plot_vert_boxplot(permutation_res: sklearn_bunch,
                      columns: list[str],
                      whiskers_len: float = 10,
                      figsize: tuple = (20, 10),
                      ) -> mpl_figure.Figure:
    sorted_importances_idx = permutation_res.importances_mean.argsort()
    importances = pd.DataFrame(
        permutation_res.importances[sorted_importances_idx].T,
        columns=columns[sorted_importances_idx],
    )
    ax = importances.plot.box(vert=False, whis=whiskers_len)
    ax.set_title("Permutation Importances (test set)")
    ax.axvline(x=0, color="k", linestyle="--")
    ax.set_xlabel("Decrease in accuracy score")
    ax.figure.tight_layout()
    return ax


def plot_power_prediciton(df: pd.DataFrame,
                          figsize: tuple = (20, 10),
                          fontsize: int = 14) -> mpl_axes.Axes:
    plt.figure(figsize=figsize)
    pps_matrix = pps.matrix(df).loc[:,['x', 'y', 'ppscore']]
    matrix_df = pps_matrix.pivot(columns='x', index='y', values='ppscore')
    heatmap = sns.heatmap(matrix_df,
                          vmin=0, vmax=1,
                          annot=True,
                          fmt=".2f",
                          annot_kws={"fontsize": fontsize})
    heatmap.set_title('Power prediciton Heatmap', fontdict={'fontsize':16}, pad=12)
    return heatmap


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("output_folder", type=click.Path())
def run_feature_importances(input_filepath: str, output_folder: str) -> None:
    """Create new features in clipped dataset

    Args:
        input_filepath (str): path to dataset with features
        output_folder (str): folder to save pictures
    """
    start_time = time.time()
        
    # read section
    save_folder = Path(Path(".") / output_folder)
    df = pd.read_csv(input_filepath)
    (features,
     windfarm_col,
     features_exc,
     perm_importance_cfg,
     ppc_cfg,
     plot_params
     ) = get_data_config("plot_feature_importance", 
         ["features",
          "windfarm_col",
          "features_exc",
          "get_permeation_importance",
          "plot_power_prediciton",
          "plot_params",
          ]
    )
    (target, seed) = get_data_config("common", ["target", "seed"])
    
    # target is last column
    df = (df.loc[:, features + [target]] if isinstance(features, list) else 
          (df.loc[:, df.columns.drop(target).to_list() + [target]]
           .drop(features_exc, axis="columns")))
    df.dropna(inplace=True, axis="index")
    df[windfarm_col] = df[windfarm_col].str.slice(start=2).astype(int)
    
    # TODO: make several if's using dict and list of required plots
    if plot_params["set_plot_params"]:
        set_plot_params()
    if perm_importance_cfg["is_on"]:
        permeation_res = get_permeation_importance(
            df,
            target,
            perm_importance_cfg["train_test_cfg"],
            perm_importance_cfg["random_forest_cfg"],
            perm_importance_cfg["n_repeats"],
            seed)
        importance_boxplot = plot_vert_boxplot(permeation_res,
                                               df.columns,
                                               ppc_cfg["whiskers_len"],
                                               plot_params["figsize"])
        save_plot(importance_boxplot, save_folder / "pfi_perm_feature_importances.png")
    if ppc_cfg["is_on"]:
        pps_heatmap = plot_power_prediciton(df,
                                            plot_params["figsize"],
                                            plot_params["fontsize"])
        save_plot(pps_heatmap, save_folder / "pfi_pps_heatmap.png")

    print(f"Execution time is {round((time.time() - start_time) / 60, 2)} minutes")

if __name__ == "__main__":
    run_feature_importances()  # pylint: disable=no-value-for-parameter
    