#!/usr/bin/env python

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

from scripts.supervised_model_utils import score_model, load_model, prepare_data_for_model, load_model_params, \
    add_supervised_target


def find_production_dates(model_params):
    sup_df = prepare_data_for_model(country=model_params["country"],
                                    mode="production",
                                    resampling_method=model_params["resampling_method"],
                                    variables=model_params["variables"],
                                    hm_days=model_params["hm_days"],
                                    functions=model_params["functions"],
                                    day_windows=model_params["day_windows"],
                                    verbose=0)

    start = sup_df.index.min()
    end = sup_df.index.max()

    return start, end, sup_df


def evaluate_production_performance(df, hm_days):
    df = add_supervised_target(df, hm_days=hm_days)

    df = df.iloc[:-hm_days]

    return df["target"]


def plot_production_results(true, preds):
    fig, ax = plt.subplots(1, 2, figsize=(20, 6))
    ax[0].set_title("true vs pred, production data")
    ax[0].plot(preds, label="preds")
    ax[0].plot(true, label="true")
    ax[0].set_ylabel("revenue")
    ax[0].legend()

    ax[1].set_title("absolute error, production data")
    ax[1].plot(abs(preds - true))
    ax[1].set_ylabel("error")
    plt.show()


def compute_production_error(true, preds):
    prod_mae = mean_absolute_error(true, preds)
    prod_rmse = mean_squared_error(true, preds, squared=False)

    prod_error = np.mean([prod_mae, prod_rmse])
    prod_error = round(prod_error, 2)

    return prod_error


def monitor(country, hm_days):
    model, model_name = load_model(country_name=country)
    model_params = load_model_params(model_name)

    start, end, sup_df = find_production_dates(model_params)

    starting_dates = list(pd.date_range(start, end))

    prod_preds = score_model(starting_dates, model_name, test=False, mode="production")

    prod_true = evaluate_production_performance(sup_df, hm_days)
    cleaned_prod_preds = pd.Series(prod_preds[:-hm_days],
                                   index=prod_true.index).apply(lambda x: x[0])

    prod_error = compute_production_error(prod_true, cleaned_prod_preds)

    print(f"Error on production data: {prod_error}")

    plot_production_results(prod_true, cleaned_prod_preds)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='monitor production data')
    parser.add_argument('-c', '--country', required=True, help='name of the country or None')
    parser.add_argument('-d', '--hm_days', default=30, type=int, help='how many days in the future to predict')

    args = parser.parse_args()

    monitor(args.country, args.hm_days)
