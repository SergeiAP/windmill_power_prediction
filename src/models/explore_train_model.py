# pylint: disable=missing-module-docstring
# TODO: change print to logs
# TODO: split file
# TODO: add adjusted model names in MLFlow 
import copy
import json
import os
import pickle  # nosec B403
import time
from pathlib import Path
from typing import Iterable

import click
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import plotly.express as px
from dotenv import load_dotenv
from matplotlib.figure import Figure as mpl_figure
from mlflow.models.signature import infer_signature
from plotly.graph_objs._figure import Figure as plotly_Figure
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import (GridSearchCV, ShuffleSplit, 
                                     cross_validate, learning_curve)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.read_config import get_data_config
from src.visualization.plot_utils import save_plot, set_plot_params


def linear_model(df_cols: list[str], 
                 cat_cols: list[str], 
                 random_state: int, 
                 **model_params
                 ) -> TransformedTargetRegressor:
    """
    Create linear model pipeline with `cat`, `num` preprocessing and `SelectFromModel`

    Args:
        df_cols (list[str]): dataframe features
        cat_cols (list[str]): categorical features
        random_state (int): random seed

    Returns:
        TransformedTargetRegressor: sklearn model
    """
    # TODO: implement OneLeaveGroupOut as cv
    
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
    model_pipe = TransformedTargetRegressor(regressor=model_pipe,
                                            func=func,
                                            inverse_func=inverse_func)
    model_pipe.set_params(**model_params) # set params for the model
    
    return model_pipe

def func(vals: np.ndarray) -> np.ndarray:
    """Target transform function for TransformedTargetRegressor

    Args:
        vals (np.ndarray): tartet col

    Returns:
        np.ndarray: transformed tagret
    """
    return np.log(vals + 0.1)

def inverse_func(vals: np.ndarray) -> np.ndarray:
    """Reverse target transform function for TransformedTargetRegressor

    Args:
        vals (np.ndarray): transformed target col

    Returns:
        np.ndarray: reverse transform target
    """
    return np.exp(vals) - 0.1


def train_model(model: TransformedTargetRegressor,
                data: pd.DataFrame,
                target: pd.Series,
                best_params: dict[str, str | float],
                metrics_dict: dict[str, dict[str, float]],
                ) -> tuple[TransformedTargetRegressor,
                           dict[str, dict[str, float]],
                           pd.DataFrame]:
    """Train model without testing on full dataset

    Args:
        model (TransformedTargetRegressor): model to be trained
        data (pd.DataFrame): data for training
        target (pd.Series): target
        best_params (dict[str, str  |  float]): params for `model`

    Returns:
        tuple[TransformedTargetRegressor, pd.DataFrame]: trained model and `coefs` for 
        linear model
    """
    intercept = 0 # by default
    
    model.set_params(**best_params)
    model.fit(data, target)
    mae_ = round(mean_absolute_error(target, model.predict(data)), 2)
    
    metrics_dict["train"] = {"MAE": mae_}
    print(f"Train MAE is: {mae_ :.2f}")
    # For linear models only
    
    if hasattr(model.regressor_[-1], 'intercept_'):  # type: ignore
        intercept = model.regressor_[-1].intercept_  # type: ignore
    if hasattr(model.regressor_[-1], 'coef_'):       # type: ignore
        feature_names = np.append(
            model.regressor_[:-1].get_feature_names_out(),         # type: ignore
            ["intercept"])
        coefs = pd.DataFrame(np.append(model.regressor_[-1].coef_, # type: ignore
                                       [intercept]), 
                            columns=['coefficients'], index=feature_names)
        coefs.index.name = "feature_name" 
        return model, metrics_dict, coefs
    return model, metrics_dict, pd.DataFrame(None)


def param_search(data: pd.DataFrame,
                 target: pd.Series,
                 model: TransformedTargetRegressor,
                 n_jobs: int,
                 grid_params: dict,
                 cv_params: dict,
                 random_state: int) -> dict:
    """Find best params for the `model` to be fitted then

    Args:
        data (pd.DataFrame): data with features for searhing
        target (pd.Series): target
        model (TransformedTargetRegressor): sklearn pipeline model
        n_jobs (int): number of jobs/processors to be used in searching, 
        -1 - all procesors
        grid_params (dict): params for searching in `GridSearchCV`
        cv_params (dict): params to be used in cross validation
        random_state (int): random seed

    Returns:
        dict: cross_validation results
    """
    # TODO: change grid search to RandomSearch 
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
                                n_jobs=n_jobs)
    cv_results = cross_validate(regressor_cv,
                                data, 
                                target,
                                return_estimator=True, 
                                cv=outer_cv, 
                                n_jobs=n_jobs,
                                **cv_params)
    
    print(f"Execution time is {round((time.time() - start_time) / 60, 2)} minutes")
    return cv_results

def retrieve_grid_params(param_grid: dict) -> dict:
    """Prepare grid search params for seaching best model params

    Args:
        param_grid (dict): params to be retrieved

    Returns:
        dict: retrieved params prepared to be used in `GridSearchCV`
    """
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

def explore_param_search(cv_results: dict,
                         key_scoring: str,
                         all_scores: Iterable[str],
                         metrics_to_track: dict[str, list[str]],
                         ) -> tuple[pd.DataFrame,
                                    dict[str, dict[str, float]],
                                    dict[str, str | float]]:
    """Explore params of inner and outer cross-validations

    Args:
        cv_results (dict): raw nested cross-validation results
        key_scoring (str): scoring for `best_params` selection
        all_scores (Iterable[str]): names of all scores to be stored
        metrics_to_track (dict[str, list[str]]): metrics to be tracked in dvc/mlflow

    Returns:
        tuple[pd.DataFrame, dict[str, str | float]]: all params params and key metrics
        and best params combination
    """
    # pepare dict for storing values
    grid_params = [key for key in cv_results["estimator"][0].best_params_.keys()]
    cv_exp = {key: [] for key in grid_params}
    cv_exp["index"] = []
    cv_exp[f"{key_scoring}_splits"] = []
    for score in all_scores:
        cv_exp[score] = [] 

    cv_exp = prepare_param_serach(cv_exp, cv_results, all_scores, key_scoring)

    # calc statistics for best models 
    cv_exp[f"{key_scoring}_median"] = [
        round(0.5 * np.median(vals) + 0.5 * metric_, 5)
        for metric_, vals in zip(cv_exp[key_scoring], cv_exp[f"{key_scoring}_splits"])]
    cv_exp[f"{key_scoring}_std"] = [
        round(np.std(vals + [metric_]), 5)
        for metric_, vals in zip(cv_exp[key_scoring], cv_exp[f"{key_scoring}_splits"])]
    
    # choose best model
    df_cv_exp = pd.DataFrame(cv_exp)
    # eval by model floor: median + std
    estimate_val = df_cv_exp[f"{key_scoring}_median"] + cv_exp[f"{key_scoring}_std"]
    df_cv_exp[f"{key_scoring}_eval"] = estimate_val
    idx_min = df_cv_exp[f"{key_scoring}_eval"].argmin()
    best_params = df_cv_exp.loc[idx_min, grid_params].to_dict()
    metrics_dict = df_metrcis_to_dict(df_cv_exp.loc[idx_min], # type: ignore
                                      metrics_to_track["test"],
                                      "test")

    print(f"Best characteristics evaluated by column {key_scoring}_eval is")
    print(df_cv_exp.loc[idx_min])
    
    return df_cv_exp, metrics_dict, best_params

def prepare_param_serach(cv_params: dict,
                         cv_results: dict,
                         all_scores: Iterable[str],
                         key_scoring: str) -> dict:
    """Extract all required params and metrics from `cv_results`

    Args:
        cv_params (dict): cross-validation 
        cv_results (dict): cross-validation results to extract required params 
        all_scores (Iterable[str]): all scores in `outer_cv` to be stored
        key_scoring (str): scoring to be used for best model selection

    Returns:
        dict: prepared params and metrics from `cv_results`
    """
    # gather best models
    for idx, estimator in enumerate(cv_results["estimator"]):
        for param, val in estimator.best_params_.items():
            cv_params[param].append(val)
        cv_params["index"].append(estimator.best_index_)
        for score in all_scores:
            cv_params[score].append(round(-1 * cv_results[score][idx], 5))
    
    # gather all scores for best models
    for idx in cv_params["index"]:
        inner_scores = []
        for estimator in cv_results["estimator"]:
            score_val = round(-1 * estimator.cv_results_["mean_test_score"][idx], 5)
            inner_scores.append(score_val)
        cv_params[f"{key_scoring}_splits"].append(inner_scores)
    return cv_params

def df_metrcis_to_dict(df_best: pd.Series,
                       metrics: list[str],
                       key: str,
                       sep_: str = "_"
                       ) -> dict[str, dict[str, float]]:
    """Convert dataframe to dict to save it as .json as metrics for dvc/mlflow

    Args:
        df_best (pd.Series): series to choose metrics
        metrics (list[str]): whihc metrics to choose
        key (str): common name for metrics
        sep_ (str, optional): separation for key and metrics. Defaults to "_".

    Returns:
        dict[str, dict[str, float]]: metrics in required  json-alike format
    """    
    metrics_dict: dict[str, dict[str, float]] = {}
    metric_names: list[str] = [key + sep_ + metric_ for metric_ in metrics]
    df_best = df_best.loc[metric_names]
    df_best.rename(lambda x: x[len(key + sep_):], inplace=True)
    metrics_dict[key] = df_best.to_dict()
    return metrics_dict
    

def plot_param_search(df_cv_exp: pd.DataFrame,
                      params_search_cfg: dict,
                      hover_cols: list[str],
                      ) -> plotly_Figure:
    """Plot parameters search results with stabdart deviations and key parameters

    Args:
        df_cv_exp (pd.DataFrame): data to be plotted
        params_search_cfg (dict): config for the plot
        hover_cols (list[str]): columns and vals to be shown in plotly when hover at
        point 

    Returns:
        plotly_Figure: plot with all info
    """
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
                        model: TransformedTargetRegressor,
                        best_params: dict[str, str | float],
                        plot_learning_curve_cfg: dict,
                        random_state: int,
                        ) -> tuple[mpl_figure, pd.DataFrame]:
    """
    Plot sklearn.model_selection.learning_curve based on fiited inside of the method
    models

    Args:
        data (pd.DataFrame): data with features
        target (pd.Series): target
        model (TransformedTargetRegressor): model to be fitted and to plot learning
        curve
        best_params (dict[str, str  |  float]): best model params to be used
        plot_learning_curve_cfg (dict): config for learning_curve
        random_state (int): seed

    Returns:
        mpl_figure: matplotlib figure - learning curve
    """
    n_splits = plot_learning_curve_cfg["n_splits"]
    scoring = plot_learning_curve_cfg["scoring"]
    figsize = plot_learning_curve_cfg["figsize"]
    n_jobs = plot_learning_curve_cfg["n_jobs"]
    
    outer_cv = ShuffleSplit(n_splits=n_splits, random_state=random_state)
    train_sizes = np.linspace(0.1, 1.0, num=n_splits, endpoint=True)
    format_dict = {"capsize": 10, "fmt": "-o", "alpha": 0.5, "elinewidth": 2}

    model.set_params(**best_params)
    results = learning_curve(model, data, target,
                            train_sizes=train_sizes,
                            cv=outer_cv,
                            scoring=scoring,
                            n_jobs=n_jobs)
    train_size, train_scores, test_scores = results[:3]
    
    # Convert the scores into errors
    train_errors, test_errors = -1 * train_scores, -1 * test_scores
    train_err_mean, train_err_std = train_errors.mean(axis=1), train_errors.std(axis=1)
    test_err_mean, test_err_std = test_errors.mean(axis=1), test_errors.std(axis=1)
    colnames = ["train_size", "train_mean", "train_std", "test_mean", "test_std"]
    metrics = [train_size, train_err_mean, train_err_std, test_err_mean, test_err_std]
    df_lc = pd.DataFrame({col: vals for col, vals in zip(colnames, metrics)})
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.errorbar(train_size, train_err_mean,
                 yerr=train_err_std, label="Training error", **format_dict)
    ax.errorbar(train_size, test_err_mean,
                 yerr=test_err_std, label="Testing error", **format_dict)
    ax.set_xlabel("Number of samples in the training set")
    ax.set_ylabel(f"{scoring[4:]}")  # del "neg_"
    ax.set_title("Learning curve for linear model")
    plt.legend()
    
    return fig, df_lc


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
    target_name: str
    best_params: dict[str, str | float]
    metrics_dict: dict[str, dict[str, float]] = {}
    df_lc: pd.DataFrame = pd.DataFrame([])
        
    # read section
    save_folder = Path(Path(".") / output_folder)
    model_folder = Path(Path(".") / model_path)
    exploratory_folder = "./reports/figures/"
    df = pd.read_csv(input_filepath)
    (features,
     set_plot_params_cfg,
     linear_model_cfg,
     params_search_cfg,
     plot_learning_curve_cfg,
     mlflow_config_cfg) = get_data_config("explore_train_model", 
         ["features",
          "set_plot_params",
          "linear_model",
          "params_search",
          "plot_learning_curve",
          'mlflow_config']
    )
    best_params = linear_model_cfg["model_params"].copy()
    scoring_names = ["test_" + metric_
                     for metric_ in params_search_cfg["cv_params"]["scoring"].keys()]
    (target_name, seed, date_col) = get_data_config("common",
        ["target", "seed", "date_col"]
    )
    # chnage for CI/CD
    if os.getenv("MLFLOW_TRACKING_URI"):
        mlflow_config_cfg["experiment_name"] = os.getenv("MLFLOW_EXPERIMENT_NAME")
    (mlflow_description, mlflow_tags, experiment_name) = (
        mlflow_config_cfg["mlflow_description"],
        mlflow_config_cfg["mlflow_tags"],
        mlflow_config_cfg["experiment_name"])
    load_dotenv()
    remote_server_uri = os.getenv("MLFLOW_TRACKING_URI")
    mlflow.set_tracking_uri(remote_server_uri) # type: ignore
    if set_plot_params_cfg["is_on"]:
        set_plot_params()
    mlflow.set_experiment(experiment_name) # experiment_id
    
    with mlflow.start_run(description=mlflow_description, tags=mlflow_tags):
        print("MLFLOW artifact uri:", mlflow.get_artifact_uri())
        # prepare dataset
        df = df.loc[(df[date_col] >= features["date_period"][0])
                    & (df[date_col] <= features["date_period"][1]), :]
        df_target: pd.Series = df.loc[:, target_name]
        df = (df.loc[:, features["include"]] if isinstance(features["include"], list)
              else (df.loc[:, df.columns.drop(target_name).to_list()]))
        data = df.drop(features["exclude"], axis="columns")
        # log params for MLFLOW
        log_params = {"seed": seed,
                    "rows_num": len(data),
                    "date_start": features["date_period"][0],
                    "date_end": features["date_period"][1]}
        
        # create model
        model = linear_model(data.columns.to_list(),
                            linear_model_cfg["cat_cols"],
                            seed,
                            **linear_model_cfg["model_params"])
        
        if params_search_cfg["is_on"]:
            print("Params search starting")
        
            cv_results = param_search(data, df_target, model,
                                    params_search_cfg["n_jobs"],
                                    params_search_cfg["grid_params"],
                                    params_search_cfg["cv_params"],
                                    seed)
            df_cv_exp, metrics_dict, best_params = explore_param_search(
                cv_results,
                params_search_cfg["plot_param_search"]["scoring"],
                scoring_names,
                params_search_cfg["plot_param_search"]["metrics_to_track"],
                )
            df_cv_exp.to_csv(save_folder / "cv_result.csv", index=False)
            print(f"Save cross-validation exploration results "
                f"as {save_folder / 'cv_result.csv'}")
            
            cv_plot = plot_param_search(df_cv_exp, params_search_cfg, [*best_params])
            save_plot(cv_plot, save_folder / "cv_plot.html") 
        
        if plot_learning_curve_cfg["is_on"]:
            print("Learning curve starting")
            
            lc_plot, df_lc = plot_learning_curve(data, df_target, model,
                                                best_params,
                                                plot_learning_curve_cfg,
                                                seed)
            save_plot(lc_plot, save_folder / "lc_plot.png")
        
        # last train of the model
        model, metrics_dict, coefs = train_model(model,
                                                 data,
                                                 df_target,
                                                 best_params,
                                                 metrics_dict)
        
        # save section
        df_lc.to_csv(model_folder / "lc_plot.csv", index=False)
        df_lc.to_csv(model_folder / "lc_plot_.csv", index=False)
        print(f"Save learning curve params as {model_folder / 'lc_plot.csv'}")
        coefs.to_csv(save_folder / "lm_coefs.csv")
        print(f"Save linear model coefficients as {save_folder / 'lm_coefs.csv'}")
        with open(model_folder / "metrics.json", 'w', encoding='utf-8') as mets_json:
            json.dump(metrics_dict, mets_json, indent=4)
        print(f"Save metrics as {save_folder / 'metrics.json'}")
        with open(model_folder / "lm_model.pkl",'wb') as file_:
            pickle.dump(model, file_)
        print(f"Save model as {model_path}/lm_model.pkl")
        
        # log for mlflow
        # signature store metadata for input-output of the model in mlflow
        signature = infer_signature(data, df_target)
        # mlflow log best params
        mlflow.log_params(best_params)
        for key_, val_ in log_params.items():
            mlflow.log_param(key_, val_)
        # mlflow log metric score
        mlflow_metric = {stage_ + "_" + metric_: vals 
                        for stage_ in metrics_dict 
                        for metric_, vals in metrics_dict[stage_].items()}
        mlflow.log_metrics(mlflow_metric)
        mlflow.log_artifact(output_folder)
        mlflow.log_artifact(exploratory_folder)
        mlflow.sklearn.log_model(sk_model=model,
                                 artifact_path=experiment_name,
                                 registered_model_name="wpp_selector-ridge",
                                 signature=signature)

    print(f"Execution time is {round((time.time() - start_time) / 60, 2)} minutes")

if __name__ == "__main__":
    run_tuning_eval_train()  # pylint: disable=no-value-for-parameter
