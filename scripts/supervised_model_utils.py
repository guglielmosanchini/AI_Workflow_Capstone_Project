import time
import joblib
import os
import sys
from warnings import simplefilter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.exceptions import ConvergenceWarning
from tqdm import tqdm
import json

from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import ParameterGrid, GridSearchCV

from scripts import ROOT_DIR
from scripts.utils import ts_conversion
from scripts.logger import update_train_log, update_predict_log


DATA_DIR = os.path.join(ROOT_DIR, "data", "datasets")
MODEL_DIR = os.path.join(ROOT_DIR, "data", "models")
TRAINING_DF_PATH = os.path.join(DATA_DIR, "df_training.csv")
PRODUCTION_DF_PATH = os.path.join(DATA_DIR, "df_production.csv")


if not sys.warnoptions:
    simplefilter("ignore", category=ConvergenceWarning)
    os.environ["PYTHONWARNINGS"] = 'ignore'


def resample_df(df, how, verbose=0):
    assert how in ("linear", "ffill", "bfill"), "how parameter must be one of 'linear', 'ffill', 'bfill'."

    if verbose > 0:
        missing_dates = set(pd.date_range(df.index.min(), df.index.max())).difference(set(df.index))
        lmd = len(missing_dates)
        print(f"There are {lmd} missing dates.")

    if how == "linear":
        # linear interpolation for missing dates
        resampled_df = df.resample('D').mean().interpolate()  # if data is daily, .mean() is redundant

    else:

        resampled_df = df.resample('D').mean().fillna(method=how)

    return resampled_df


def add_supervised_target(df, hm_days):
    result_df = df.copy()

    target = []

    for day in result_df.index:
        start = day + pd.Timedelta(days=1)
        end = start + pd.Timedelta(days=hm_days)

        rev_next_days = df["revenue"].loc[start:end].sum()

        target.append(rev_next_days)

    result_df["target"] = target

    return result_df


def convert_to_supervised(df, variables=("revenue", "invoices"), hm_days=30, functions=("mean",),
                          day_windows=(3, 5, 7)):
    assert all(v in df.columns for v in variables), "variables must be a tuple of columns of the input dataframe."

    df_with_target = add_supervised_target(df, hm_days)

    df_with_new_variables = add_supervised_variables(df, variables, functions, day_windows)

    supervised_df = pd.merge(df_with_target, df_with_new_variables, on="date")

    return supervised_df


def add_supervised_variables(df, variables, functions=("mean",), day_windows=(3, 5, 7)):
    func_names = {"mean": np.mean, "std": np.std, "var": np.var, "sum": np.sum}

    assert all([f in func_names.keys() for f in functions]), "Acceptable functions are: 'mean', 'std', 'var', 'sum'."

    variables = list(variables)

    # build rolling means/stds/vars/sums for variables with input window days
    df_with_new_vars = pd.DataFrame(index=df.index)
    for dw in day_windows:
        for func_name in functions:
            temp_df = df[variables].rolling(dw).apply(func_names[func_name])
            temp_df.columns = [col + "_" + str(dw) + "_" + func_name for col in temp_df.columns]
            df_with_new_vars = df_with_new_vars.merge(temp_df, on="date")

    # drop rows with NaNs
    df_with_new_vars = df_with_new_vars.dropna()

    return df_with_new_vars


def prepare_data_for_model(country, mode, resampling_method="linear", variables=("revenue", "invoices"), hm_days=30,
                           functions=("mean",), day_windows=(3, 5, 7), verbose=0):
    assert mode in ("train", "test", "production"), "mode parameter must be 'train', 'test' or 'production'."

    if mode != "production":
        original_df = pd.read_csv(TRAINING_DF_PATH)
    else:
        original_df = pd.read_csv(PRODUCTION_DF_PATH)

    df = ts_conversion(original_df, country)

    resampled_df = resample_df(df, resampling_method, verbose)

    if mode == "train":
        sup_df = convert_to_supervised(resampled_df, variables, hm_days, functions, day_windows)
    elif mode in ("test", "production"):
        sup_df = add_supervised_variables(resampled_df, variables, functions, day_windows)
        sup_df = pd.merge(sup_df, resampled_df, on="date")

    return sup_df


def split_train_test(df, training_perc=0.8, hm_days=30, verbose=0):
    first_date_training = df.index.min()
    last_date_training = first_date_training + pd.Timedelta(days=int(len(df) * training_perc))
    first_date_testing = last_date_training + pd.Timedelta(days=1)
    last_date_testing = df.index.max()

    training_data = df.loc[first_date_training:last_date_training, :]
    testing_data = df.loc[first_date_testing:last_date_testing, :]

    if verbose > 0:
        print(first_date_training, last_date_training, first_date_testing, last_date_testing)

    # correction for the supervised problem: shrink dataset by removing the last hm_days rows,
    # because future revenue data is insufficient to compute the next hm_days days of revenue
    training_data = training_data.iloc[:-hm_days].copy()
    testing_data = testing_data.iloc[:-hm_days].copy()

    if verbose > 0:
        print(training_data.shape, testing_data.shape)

    target_column = "target"
    X_train, y_train = training_data.drop(target_column, 1).values, training_data[target_column].values
    X_test, y_test = testing_data.drop(target_column, 1).values, testing_data[target_column].values

    X = np.concatenate([X_train, X_test])
    y = np.concatenate([y_train, y_test])

    return X, y, X_train, y_train, X_test, y_test


def find_best_model(scaler, algorithm, X_train, y_train, cv=5, verbose=0):
    assert scaler in ("standard", "minmax", None), "scaler must be either None or one of 'standard', 'minmax'."

    assert algorithm in ("random_forest", "mlp"), "algorithm must be 'random_forest' or 'mlp'."

    if algorithm == "random_forest":
        alg = RandomForestRegressor()

        param_grid = {
            'alg__n_estimators': [50, 100, 200, 500],
            'alg__criterion': ["mse", "mae"]
        }
    elif algorithm == "mlp":
        alg = MLPRegressor()

        param_grid = {
            'alg__hidden_layer_sizes': [(30,), (30, 10), (50,)],
            'alg__solver': ["lbfgs"],
            'alg__alpha': [0.001, 0.0001],
            'alg__max_iter': [5_000],
            'alg__activation': ['relu', 'tanh']
        }

    if scaler is not None:
        if scaler == "standard":
            scaler = StandardScaler()
        elif scaler == "minmax":
            scaler = MinMaxScaler()
        pipe = Pipeline([('scaler', scaler), ('alg', alg)])
    else:
        pipe = Pipeline([('alg', alg)])

    # Parameters of pipelines can be set using ‘__’ separated parameter names:
    search = GridSearchCV(pipe, param_grid, n_jobs=-1, cv=cv, verbose=verbose)

    search.fit(X_train, y_train)

    if verbose > 0:
        print("Best parameter (CV score = %0.3f):" % search.best_score_)
        print(search.best_params_)

    model = search.best_estimator_

    return model


def evaluate_model(model, X_test, y_test, plot_eval, plot_avg_threshold=np.inf):
    test_predictions = model.predict(X_test)

    test_mae = round(mean_absolute_error(test_predictions, y_test), 2)
    test_rmse = round(mean_squared_error(test_predictions, y_test, squared=False), 2)
    test_avg = round(np.mean([test_mae, test_rmse]), 2)

    if (plot_eval is True) & (test_avg < plot_avg_threshold):
        fig, ax = plt.subplots(1, 1, figsize=(14, 6))
        title = f"{model.named_steps} \n test_mae: {test_mae}, test_rmse:{test_rmse}, test_avg:{test_avg}"
        ax.set_title(title)
        ax.plot(test_predictions, label="pred")
        ax.plot(y_test, label="true")
        ax.legend()
        plt.show(block=False)

    return test_mae, test_rmse, test_avg


def create_train_test(country, resampling_method="linear", variables=("revenue", "invoices"), hm_days=30,
                      functions=("mean",), day_windows=(3, 5, 7), training_perc=0.8, verbose=0):
    sup_df = prepare_data_for_model(country, "train", resampling_method, variables, hm_days, functions, day_windows,
                                    verbose)

    X, y, X_train, y_train, X_test, y_test = split_train_test(sup_df, training_perc, hm_days, verbose)

    return X, y, X_train, y_train, X_test, y_test


def build_and_eval_supervised_model(country, resampling_method="linear", variables=("revenue", "invoices"), hm_days=30,
                                    functions=("mean",), day_windows=(3, 5, 7), training_perc=0.8, scaler="standard",
                                    algorithm="random_forest", cv=5, plot_eval=False, plot_avg_threshold=np.inf,
                                    verbose=0):
    X, y, X_train, y_train, X_test, y_test = create_train_test(country, resampling_method, variables,
                                                               hm_days, functions, day_windows, training_perc)

    model = find_best_model(scaler, algorithm, X_train, y_train, cv, verbose)

    model.fit(X, y)

    test_mae, test_rmse, test_avg = evaluate_model(model, X_test, y_test, plot_eval, plot_avg_threshold)

    return model, test_mae, test_rmse, test_avg


def param_grid_selector(param_dim):
    # for testing only
    if param_dim == "very_small":
        gs_params = {
            'variables': [('revenue', 'invoices')],
            'scaler': [None],
            'algorithm': ['random_forest', 'mlp'],
            'resampling_method': ['linear'],
            'functions': [('mean',)],
            'day_windows': [(3, 5, 7)]
        }

    elif param_dim == "small":
        gs_params = {
            'variables': [(),
                          ('revenue', 'invoices')],
            'scaler': ['minmax', None],
            'algorithm': ['random_forest', 'mlp'],
            'resampling_method': ['linear'],
            'functions': [('mean',)],
            'day_windows': [(3,), (3, 5, 7), (7, 14)]
        }

    elif param_dim == "medium":
        gs_params = {
            'variables': [(),
                          ('revenue',),
                          ('revenue', 'invoices', 'purchases')],
            'scaler': ['standard', 'minmax', None],
            'algorithm': ['random_forest', 'mlp'],
            'resampling_method': ['linear'],
            'functions': [('mean',), ('mean', 'std')],
            'day_windows': [(3,), (7,), (3, 5), (3, 5, 7), (7, 14)]
        }

    elif param_dim == "large":
        gs_params = {
            'variables': [(),
                          ('revenue',),
                          ('invoices',),
                          ('revenue', 'invoices'),
                          ('revenue', 'invoices', 'purchases')],
            'scaler': ['standard', 'minmax', None],
            'algorithm': ['random_forest', 'mlp'],
            'resampling_method': ['linear', 'ffill'],
            'functions': [('mean',), ('mean', 'std')],
            'day_windows': [(3,), (5,), (7,), (3, 5), (3, 5, 7), (7, 14), (3, 5, 7, 14)]
        }

    return gs_params


def grid_search_pipeline(country, cv=2, param_dim="small", plot_if_better=False, hm_days=30):
    assert param_dim in ("very_small", "small", "medium", "large"), "param_dim must be one of 'very_small', 'small', 'medium', 'large'."

    results = {}
    best_model = None
    best_params = None

    gs_params = param_grid_selector(param_dim)

    param_grid = ParameterGrid(gs_params)

    print(f"n_combinations: {len(param_grid)}")

    plot_avg_threshold = np.inf

    for i, pg in enumerate(tqdm(param_grid)):

        model, test_mae, test_rmse, test_avg = build_and_eval_supervised_model(country=country,
                                                                               resampling_method=pg[
                                                                                   "resampling_method"],
                                                                               variables=pg["variables"],
                                                                               hm_days=hm_days,
                                                                               functions=pg["functions"],
                                                                               day_windows=pg["day_windows"],
                                                                               training_perc=0.8,
                                                                               scaler=pg["scaler"],
                                                                               algorithm=pg["algorithm"],
                                                                               cv=cv,
                                                                               plot_eval=plot_if_better,
                                                                               plot_avg_threshold=plot_avg_threshold,
                                                                               verbose=0)
        pg["hm_days"] = hm_days
        pg["country"] = country
        pg["cv"] = cv

        if test_avg < plot_avg_threshold:
            plot_avg_threshold = test_avg
            best_model = model
            best_params = pg

        errors = {"test_mae": test_mae,
                  "test_rmse": test_rmse,
                  "avg": test_avg}
        results[i] = {"params": pg, "errors": errors}

    sorted_results_by_avg = sorted(results.items(), key=lambda x_y: x_y[1]['errors']['avg'])

    return best_model, best_params, sorted_results_by_avg


def save_model(model, model_params, model_name):
    model_name = model_name.replace(".joblib", "")

    # find the model version name
    all_files_in_models = os.listdir(MODEL_DIR)
    all_model_names = [file.replace(".joblib", "") for file in all_files_in_models if file.endswith(".joblib")]
    version_numbers = [int(_model_name.split("_")[-1]) for _model_name in all_model_names]
    if len(version_numbers) == 0:
        new_version_number = "0"
    else:
        new_version_number = str(max(version_numbers) + 1)

    model_name = model_name + "_" + new_version_number

    model_saving_path = os.path.join(MODEL_DIR, model_name + ".joblib")
    params_saving_path = os.path.join(MODEL_DIR, model_name + ".json")

    # save model
    joblib.dump(model, model_saving_path)

    # save model params
    with open(params_saving_path, 'w') as f:
        json.dump(model_params, f)

    print(f"Model and params saved in {MODEL_DIR}.")

    return model_name, new_version_number


def load_model(model_name=None, country_name=None):

    if model_name is None:
        model_name = find_last_model(country_name)
        print(model_name)

    model_name = model_name.replace(".joblib", "")
    model_loading_path = os.path.join(MODEL_DIR, model_name + ".joblib")

    # load model
    loaded_model = joblib.load(model_loading_path)

    return loaded_model, model_name


def load_model_params(model_name):
    model_name = model_name.replace(".joblib", "")
    params_loading_path = os.path.join(MODEL_DIR, model_name + ".json")

    # load params
    with open(params_loading_path) as f:
        loaded_params = json.load(f)

    # they are saved as lists, because JavaScript uses arrays writtne with squared brackets
    for ll in ["day_windows", "functions", "variables"]:
        loaded_params[ll] = tuple(loaded_params[ll])

    return loaded_params


def train_model(country, param_dim="small", test=False):
    """
    Train the model using a double gridsearch (one for the data trasformation, one for the model hyperparameters)
    and save it, ready to be called.

    :param country: None or a name of a country where to train the model.
    :param param_dim: 'very_small', 'small', 'medium' or 'large', indicates the dimension of the parameter grid.
    :param test: if True, the training is being done for testing purposes.
    :return: saved model name.
    """

    start_time = time.time()

    if country == "null":
        country = None

    best_model, best_params, r = grid_search_pipeline(country, param_dim=param_dim)
    best_avg_score = r[0][1]["errors"]["avg"]

    print(f"Best score on test set: {best_avg_score}")

    model_name = f"supervised_model_{param_dim}_{country}"
    if test is False:
        model_name, model_version = save_model(best_model, best_params, model_name)
    else:
        model_version = "000"
        model_name = model_name + "_" + model_version + ".joblib"

    end_time = time.time()
    runtime = end_time-start_time

    update_train_log(best_params, best_avg_score, param_dim, runtime, model_version, test)

    return model_name


def score_model(starting_dates, model_name, test=False, mode="test"):

    assert mode in ("test", "production"), "mode must be 'test' or 'production'"

    start_time = time.time()

    #if model_name is None:
    #    model_name = find_last_model(country_name)

    # load model and params
    model, model_name = load_model(model_name)
    model_params = load_model_params(model_name)
    model_version = model_name.replace(".joblib", "").split("_")[-1]

    sup_df = prepare_data_for_model(country=model_params["country"],
                                    mode=mode,
                                    resampling_method=model_params["resampling_method"],
                                    variables=model_params["variables"],
                                    hm_days=model_params["hm_days"],
                                    functions=model_params["functions"],
                                    day_windows=model_params["day_windows"],
                                    verbose=0)

    starting_dates = [pd.Timestamp(sd) for sd in starting_dates]

    if any(sd not in sup_df.index for sd in starting_dates):
        start = sup_df.index.min().strftime("%Y-%m-%d")
        end = sup_df.index.max().strftime("%Y-%m-%d")
        raise KeyError(f"Acceptables dates range from {start} to {end}.")

    predictions = []
    for sd in starting_dates:
        x = sup_df.loc[sd].values
        x = x.reshape(1, -1)
        prediction = model.predict(x)
        predictions.append(prediction)

    end_time = time.time()
    runtime = end_time - start_time

    update_predict_log(predictions, starting_dates, runtime, model_version, test)

    return predictions


def compare_best_with_baseline(country, best_model_name=None):

    # take the one with highest version number
    if best_model_name is None:
        best_model_name = find_last_model()
        print(best_model_name)

    X, y, X_train, y_train, X_test, y_test = create_train_test(country)

    reg = LinearRegression()
    reg.fit(X_train, y_train)
    base_pred = reg.predict(X_test)

    best_model, best_model_name = load_model(best_model_name)
    bm_pred = best_model.predict(X_test)

    baseline_err = abs(y_test - base_pred)
    bm_err = abs(y_test - bm_pred)

    xlabs = pd.date_range("2019-04-03", "2019-07-01")

    fig, ax = plt.subplots(1, 1, figsize=(20, 8))
    ax.set_title("Baseline model vs Best model")
    ax.set_ylabel("Error")
    ax.set_xlabel("Date")

    ax.scatter(xlabs, baseline_err, s=80, alpha=1, label="baseline model error", color="purple", zorder=2)
    ax.scatter(xlabs, bm_err, s=80, alpha=1, label="best model error", color="green", zorder=2)

    for i, (be, bm) in enumerate(zip(baseline_err, bm_err)):
        if be <= bm:
            c = "red"
            lw = 3
        else:
            c = "orange"
            lw = 1
        ax.vlines(x=xlabs[i], ymin=min(be, bm), ymax=max(be, bm), lw=lw, color=c, zorder=1)

    ax.legend()
    plt.show()


def find_last_model(country_name):

    if country_name is None:
        country_name = "None"

    all_files_in_models = os.listdir(MODEL_DIR)
    all_models = [file for file in all_files_in_models if (file.endswith(".joblib") and country_name in file)]
    all_models = sorted(all_models, key=lambda x: int(x.replace(".joblib", "").split("_")[-1]))

    if len(all_models) == 0:
        raise FileNotFoundError

    best_model_name = all_models[-1]
    return best_model_name


if __name__ == "__main__":
    country = None
    param_dim = "small"
    testing_dates = ["2018-01-25"]

    print("Training the Model")
    model_name = train_model(country, param_dim)

    print(f"Testing the model on {testing_dates}")
    prediction = score_model(testing_dates, model_name=model_name)
    print(prediction)
