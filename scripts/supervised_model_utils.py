import joblib
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import json

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import ParameterGrid, GridSearchCV

from scripts.utils import ts_conversion

DATA_FOLDER = os.path.join("..", "data")
MODEL_FOLDER = os.path.join(DATA_FOLDER, "models")
TRAINING_DF_PATH = os.path.join(DATA_FOLDER, "df_training.csv")


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
    variables = list(variables)

    func_names = {"mean": np.mean, "std": np.std, "var": np.var, "sum": np.sum}

    # build rolling means for variables with input window days
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
    assert mode in ("train", "test"), "mode parameter must be 'train' or 'test'."

    original_df = pd.read_csv(TRAINING_DF_PATH)
    df = ts_conversion(original_df, country)

    resampled_df = resample_df(df, resampling_method, verbose)

    if mode == "train":
        sup_df = convert_to_supervised(resampled_df, variables, hm_days, functions, day_windows)
    elif mode == "test":
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


def find_best_model(scaler, X_train, y_train, cv=5, verbose=0):
    assert scaler in ("standard", "minmax", None), "scaler must be either None or one of 'standard', 'minmax'."

    if scaler is not None:
        if scaler == "standard":
            scaler = StandardScaler()
        elif scaler == "minmax":
            scaler = MinMaxScaler()
        pipe = Pipeline([('scaler', scaler), ('rfr', RandomForestRegressor())])
    else:
        pipe = Pipeline([('rfr', RandomForestRegressor())])

    # Parameters of pipelines can be set using ‘__’ separated parameter names:
    param_grid = {
        'rfr__n_estimators': [50, 100, 200, 500],
        'rfr__criterion': ["mse", "mae"],
    }

    search = GridSearchCV(pipe, param_grid, n_jobs=-1, cv=cv, verbose=verbose)

    search.fit(X_train, y_train)

    if verbose > 0:
        print("Best parameter (CV score = %0.3f):" % search.best_score_)
        print(search.best_params_)

    model = search.best_estimator_

    return model


def train_model(model, X, y):
    model.fit(X, y)

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

    return test_mae, test_rmse


def create_train_test(country, resampling_method="linear", variables=("revenue", "invoices"), hm_days=30,
                      functions=("mean",), day_windows=(3, 5, 7), training_perc=0.8, verbose=0):
    sup_df = prepare_data_for_model(country, "train", resampling_method, variables, hm_days, functions, day_windows,
                                    verbose)

    X, y, X_train, y_train, X_test, y_test = split_train_test(sup_df, training_perc, hm_days, verbose)

    return X, y, X_train, y_train, X_test, y_test


def build_and_eval_supervised_model(country, resampling_method="linear", variables=("revenue", "invoices"), hm_days=30,
                                    functions=("mean",), day_windows=(3, 5, 7), training_perc=0.8, scaler="standard",
                                    cv=5, plot_eval=False, plot_avg_threshold=np.inf, verbose=0):
    X, y, X_train, y_train, X_test, y_test = create_train_test(country, resampling_method, variables,
                                                               hm_days, functions, day_windows, training_perc)

    model = find_best_model(scaler, X_train, y_train, cv, verbose)

    model = train_model(model, X, y)

    test_mae, test_rmse = evaluate_model(model, X_test, y_test, plot_eval, plot_avg_threshold)

    return model, test_mae, test_rmse


def param_grid_selector(param_dim):
    if param_dim == "small":
        gs_params = {
            'variables': [(),
                          ('revenue', 'invoices'), ],
            'scaler': ['minmax', None],
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
            'resampling_method': ['linear', 'ffill'],
            'functions': [('mean',), ('mean', 'std')],
            'day_windows': [(3,), (5,), (7,), (3, 5), (3, 5, 7), (7, 14), (3, 5, 7, 14)]
        }

    return gs_params


def grid_search_pipeline(country, cv, param_dim, plot_if_better=False, hm_days=30):
    assert param_dim in ("small", "medium", "large"), "param_dim must be one of 'small', 'medium', 'large'."

    results = {}
    best_model = None
    best_params = None

    gs_params = param_grid_selector(param_dim)

    param_grid = ParameterGrid(gs_params)

    print(f"n_combinations: {len(param_grid)}")

    plot_avg_threshold = np.inf

    for i, pg in enumerate(tqdm(param_grid)):

        model, test_mae, test_rmse = build_and_eval_supervised_model(country=country,
                                                                     resampling_method=pg["resampling_method"],
                                                                     variables=pg["variables"],
                                                                     hm_days=hm_days,
                                                                     functions=pg["functions"],
                                                                     day_windows=pg["day_windows"],
                                                                     training_perc=0.8,
                                                                     scaler=pg["scaler"],
                                                                     cv=cv,
                                                                     plot_eval=plot_if_better,
                                                                     plot_avg_threshold=plot_avg_threshold,
                                                                     verbose=0)
        avg = round(np.mean([test_mae, test_rmse]), 2)
        pg["hm_days"] = hm_days
        pg["country"] = country
        pg["cv"] = cv

        if avg < plot_avg_threshold:
            plot_avg_threshold = avg
            best_model = model
            best_params = pg

        errors = {"test_mae": test_mae,
                  "test_rmse": test_rmse,
                  "avg": avg}
        results[i] = {"params": pg, "errors": errors}

    sorted_results_by_avg = sorted(results.items(), key=lambda x_y: x_y[1]['errors']['avg'])

    return best_model, best_params, sorted_results_by_avg


def save_model(model, model_params, model_name):
    model_name = model_name.replace(".joblib", "")

    # find the model version name
    all_files_in_models = os.listdir(MODEL_FOLDER)
    all_model_names = [file for file in all_files_in_models if file.endswith(".joblib")]
    version_numbers = [int(_model_name.split("_")[-1]) for _model_name in all_model_names]
    if len(version_numbers) == 0:
        new_version_number = "0"
    else:
        new_version_number = str(max(version_numbers) + 1)

    model_name = model_name + "_" + new_version_number

    model_saving_path = os.path.join(MODEL_FOLDER, model_name + ".joblib")
    params_saving_path = os.path.join(MODEL_FOLDER, model_name + ".json")

    # save model
    joblib.dump(model, model_saving_path)

    # save model params
    with open(params_saving_path, 'w') as f:
        json.dump(model_params, f)

    print(f"Model and params saved in {MODEL_FOLDER}.")


def load_model(model_name):
    model_name = model_name.replace(".joblib", "")
    model_loading_path = os.path.join(MODEL_FOLDER, model_name + ".joblib")
    params_loading_path = os.path.join(MODEL_FOLDER, model_name + ".json")

    # load model
    loaded_model = joblib.load(model_loading_path)

    # load params
    with open(params_loading_path) as f:
        loaded_params = json.load(f)

    # they are saved as lists, because JavaScript uses arrays writtne with squared brackets
    for ll in ["day_windows", "functions", "variables"]:
        loaded_params[ll] = tuple(loaded_params[ll])

    return loaded_model, loaded_params


def score_model(model, model_params, starting_date):
    sup_df = prepare_data_for_model(country=model_params["country"],
                                    mode="test",
                                    resampling_method=model_params["resampling_method"],
                                    variables=model_params["variables"],
                                    hm_days=model_params["hm_days"],
                                    functions=model_params["functions"],
                                    day_windows=model_params["day_windows"],
                                    verbose=0)

    starting_date = pd.Timestamp(starting_date)

    if starting_date not in sup_df.index:
        start = sup_df.index.min().strftime("%Y-%m-%d")
        end = sup_df.index.max().strftime("%Y-%m-%d")
        raise KeyError(f"Acceptables dates range from {start} to {end}.")

    x = sup_df.loc[starting_date].values

    x = x.reshape(1, -1)

    prediction = model.predict(x)

    return prediction


if __name__ == "__main__":

    country = None
    param_dim = "small"
    hm_days = 30
    cv = 2
    best_model, best_params, r = grid_search_pipeline(country=country,
                                                      cv=cv,
                                                      param_dim=param_dim,
                                                      plot_if_better=False,
                                                      hm_days=hm_days)

    model_name = f"supervised_model_{param_dim}_{country}"
    save_model(best_model, best_params, model_name)
