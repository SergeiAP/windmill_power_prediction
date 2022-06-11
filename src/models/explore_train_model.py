# pylint: disable=missing-module-docstring
# TODO: change print to logs
import copy
import pickle
import time
from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
from matplotlib.figure import Figure as mpl_figure
from plotly.graph_objs._figure import Figure as plotly_Figure
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import (GridSearchCV, ShuffleSplit,
                                     cross_validate, learning_curve)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.read_config import get_data_config
from src.visualization.plot_utils import (  # pylint: disable=import-error
    save_plot, set_plot_params)


def linear_model(df_cols: list[str],
                 cat_cols: list[str],
                 random_state: int,
                 **model_params
                 ) -> Pipeline:
    num_cols = [col for col in df_cols if col not in cat_cols]
    preprocessor = ColumnTransformer(
    transformers = [
        ("num", StandardScaler(), num_cols),
        ("cat",  OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ]
    )
    selector = SelectFromModel(ExtraTreesRegressor(min_samples_leaf=10,
                                                   random_state=random_state),
                               prefit=False)
    regressor = Ridge(alpha=0.5, fit_intercept=True)
    model_pipe = Pipeline(steps=[("preprocesor", preprocessor),
                                 ("feature_selector", selector),
                                 ("model", regressor)])
    model_pipe.set_params(**model_params) # set params for the model
    
    return model_pipe


def train_model(model: Pipeline,
                data: pd.DataFrame,
                target: pd.Series,
                cast_lasts: str,
                best_params: dict
                ) -> tuple[Pipeline, pd.DataFrame]:
    intercept = 0 # by default
    real_intercept = 0 # by default
    
    model.set_params(**best_params)
    model.fit(data, target)
    
    print(f"Train MAPE is: "
          f"{mean_absolute_percentage_error(model.predict(data), target) * 100:.2f}%")
    # For linear models only
    
    if hasattr(model[-1], 'intercept_'):
        real_intercept = model[-1].intercept_                            # type: ignore
        intercept = model[-1].intercept_ / data[cast_lasts].median()     # type: ignore
    if hasattr(model[-1], 'coef_'):
        feature_names = np.append(model[:-1].get_feature_names_out(),    # type: ignore
                                  ["intercept", "real_intercept"])
        coefs = pd.DataFrame(np.append(model[-1].coef_,                  # type: ignore
                                       [intercept, real_intercept]), 
                            columns=['coefficients'], index=feature_names)
        return model, coefs
    return model, pd.DataFrame(None)


def param_search(data: pd.DataFrame,
                 target: pd.Series,
                 model: Pipeline,
                 grid_params: dict,
                 cv_params: dict,
                 random_state: int) -> dict:
    grid_params = copy.deepcopy(grid_params)
    cv_params = copy.deepcopy(cv_params)
    
    start_time = time.time()
    
    inner_cv = ShuffleSplit(n_splits=grid_params.pop("n_splits", None),
                            random_state=random_state)
    outer_cv = ShuffleSplit(n_splits=cv_params.pop("n_splits", None),
                            random_state=random_state)
    inner_scoring: str = grid_params.pop("scoring", None)
    grid_params = retrieve_grid_params(grid_params)
    
    regressor_cv = GridSearchCV(estimator=model,
                                param_grid=grid_params,
                                scoring=inner_scoring,
                                cv=inner_cv,
                                n_jobs=-1)
    cv_results = cross_validate(regressor_cv, data, target,
                                return_estimator=True, cv=outer_cv, n_jobs=-1,
                                **cv_params)
    
    print(f"Execution time is {round((time.time() - start_time) / 60, 2)} minutes")
    return cv_results

def retrieve_grid_params(param_grid: dict) -> dict:
    res_param_grid = copy.deepcopy(param_grid)
    for key, elem in param_grid.items():
        match elem[0]:
            case "func":
                match elem[1][0]:
                    case "log":
                        range_ = np.logspace(elem[1][1], elem[1][2], num=elem[1][3])
                        res_param_grid[key] = range_
            case "vals":
                res_param_grid[key] = elem[1]
    return res_param_grid

def explore_param_search(cv_results: dict, scoring: str) -> tuple[pd.DataFrame, dict]:
    """"""
    # pepare dict for storing values
    grid_params = [key for key in cv_results["estimator"][0].best_params_.keys()]
    cv_exp = {key: [] for key in grid_params}
    cv_exp["index"] = []
    cv_exp[scoring], cv_exp[f"{scoring}_splits"] = [], []

    cv_exp = prepare_param_serach(cv_exp, cv_results, scoring)

    # calc statistics for best models 
    cv_exp[f"{scoring}_median"] = [round(np.median(vals), 5) 
                                    for vals in cv_exp[f"{scoring}_splits"]]
    cv_exp[f"{scoring}_std"] = [round(np.std(vals), 5) 
                                   for vals in cv_exp[f"{scoring}_splits"]]
    
    # choose best model
    df_cv_exp = pd.DataFrame(cv_exp)
    # eval by model floor: median + std
    estimate_val = df_cv_exp[f"{scoring}_median"] + cv_exp[f"{scoring}_std"]
    df_cv_exp[f"{scoring}_eval"] = estimate_val
    idx_min = df_cv_exp[f"{scoring}_eval"].argmin()
    best_params = df_cv_exp.loc[idx_min, grid_params].to_dict()

    print(f"Best characteristics evaluated by column {scoring}_eval is")
    print(df_cv_exp.loc[df_cv_exp[f"{scoring}_eval"].argmin()])
    
    return df_cv_exp, best_params


def prepare_param_serach(cv_params: dict, cv_results: dict, scoring: str) -> dict:
    # gather best models
    for estimator in cv_results["estimator"]:
        for param, val in estimator.best_params_.items():
            cv_params[param].append(val)
        cv_params["index"].append(estimator.best_index_)
        cv_params[scoring].append(round(-1 * estimator.best_score_, 5))
    
    # gather all scores for best models
    for idx in cv_params["index"]:
        inner_scores = []
        for estimator in cv_results["estimator"]:
            score_val = round(-1 * estimator.cv_results_["mean_test_score"][idx], 5)
            inner_scores.append(score_val)
        cv_params[f"{scoring}_splits"].append(inner_scores)
    return cv_params


def plot_param_search(df_cv_exp: pd.DataFrame,
                      params_search_cfg: dict,
                      hover_cols: list[str],
                      ) -> plotly_Figure:
    """"""
    y_axis_name = params_search_cfg["plot_param_search"]["scoring"] + "_median"
    x_axis_name = params_search_cfg["plot_param_search"]["x_axis"]
    error_name = params_search_cfg["plot_param_search"]["scoring"] + "_std"
    figsize = params_search_cfg["plot_param_search"]["figsize"]
    hover_cols = [*params_search_cfg["plot_param_search"]["hover_cols"], *hover_cols]
    
    fig = px.scatter(df_cv_exp, x=x_axis_name, y=y_axis_name, 
                     error_y=error_name, error_y_minus=error_name,
                     hover_data=hover_cols,
                     title="Cross validation results")
    fig.update_layout(width=figsize[0]*80, height=figsize[1]*80)
    return fig
    
    
def plot_learning_curve(data: pd.DataFrame,
                        target: pd.Series,
                        model: Pipeline,
                        best_params: dict,
                        plot_learning_curve_cfg: dict,
                        random_state: int,
                        ) -> mpl_figure:
    """"""
    n_splits = plot_learning_curve_cfg["n_splits"]
    scoring = plot_learning_curve_cfg["scoring"]
    figsize = plot_learning_curve_cfg["figsize"]
    
    outer_cv = ShuffleSplit(n_splits=n_splits, random_state=random_state)
    train_sizes = np.linspace(0.1, 1.0, num=10, endpoint=True)
    format_dict = {"capsize": 10, "fmt": "-o", "alpha": 0.5, "elinewidth": 2}

    model.set_params(**best_params)
    results = learning_curve(model, data, target,
                            train_sizes=train_sizes,
                            cv=outer_cv,
                            scoring=scoring,
                            n_jobs=-1)
    train_size, train_scores, test_scores = results[:3]
    # Convert the scores into errors
    train_errors, test_errors = -1 * train_scores, -1 * test_scores
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.errorbar(train_size, train_errors.mean(axis=1),
                 yerr=train_errors.std(axis=1), label="Training error", **format_dict)
    ax.errorbar(train_size, test_errors.mean(axis=1),
                 yerr=test_errors.std(axis=1), label="Testing error", **format_dict)
    ax.set_xlabel("Number of samples in the training set")
    ax.set_ylabel(f"{scoring[4:]}")  # del "neg_"
    ax.set_title("Learning curve for linear model")
    plt.legend()
    
    return fig


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("output_folder", type=click.Path(exists=True))
@click.argument("model_path", type=click.Path())
def run_tuning_eval_train(input_filepath: str,
                          output_folder: str,
                          model_path: str,
                          ) -> None:
    """Explore and train model

    Args:
        input_filepath (str): path to dataset with features
        output_folder (str): folder to save pictures
        model_path (str): path to save the model
    """
    start_time = time.time()
    features: dict
    target: str
        
    # read section
    save_folder = Path(Path(".") / output_folder)
    df = pd.read_csv(input_filepath)
    (features,
     set_plot_params_cfg,
     linear_model_cfg,
     params_search_cfg,
     plot_learning_curve_cfg) = get_data_config("explore_train_model", 
         ["features",
          "set_plot_params",
          "linear_model",
          "params_search",
          "plot_learning_curve"]
    )
    (target, seed, cast_lasts) = get_data_config("common",
                                                 ["target_name", "seed", "cast_lasts"])
    if set_plot_params_cfg["is_on"]:
        set_plot_params()
    
    df_target: pd.Series = df.loc[:, target]
    df = (df.loc[:, features["include"]] if isinstance(features["include"], list) else 
          (df.loc[:, df.columns.drop(target).to_list()]))
    data = df.drop(features["exclude"], axis="columns")
    
    model = linear_model(data.columns.to_list(),
                         linear_model_cfg["cat_cols"],
                         seed,
                         **linear_model_cfg["model_params"])
    
    if params_search_cfg["is_on"]:
    
        cv_results = param_search(data, df_target, model,
                                  params_search_cfg["grid_params"],
                                  params_search_cfg["cv_params"],
                                  seed)
        df_cv_exp, best_params = explore_param_search(
            cv_results, params_search_cfg["plot_param_search"]["scoring"])
        df_cv_exp.to_csv(save_folder / "cv_result.csv")
        print(f"Save cross-validation exploration results "
              f"as {save_folder / 'cv_result.csv'}")
        
        cv_plot = plot_param_search(df_cv_exp, params_search_cfg, [*best_params])
        save_plot(cv_plot, save_folder / "cv_plot.html")
    
    if plot_learning_curve_cfg["is_on"]:
        lc_plot = plot_learning_curve(data, df_target,
                                      model,
                                      best_params, # type: ignore 
                                      plot_learning_curve_cfg,
                                      seed)
        save_plot(lc_plot, save_folder / "lc_plot.png")
    
    model, coefs = train_model(model, data,
                               df_target,
                               cast_lasts, best_params) # type: ignore
    coefs.to_csv(save_folder / "lm_coefs.csv")
    print(f"Save linear model coefficients "
            f"as {save_folder / 'lm_coefs.csv'}")
    with open(model_path,'wb') as f:
        pickle.dump(model, f)
        print(f"Save model as {model_path}")
    

    print(f"Execution time is {round((time.time() - start_time) / 60, 2)} minutes")

if __name__ == "__main__":
    run_tuning_eval_train()  # pylint: disable=no-value-for-parameter
