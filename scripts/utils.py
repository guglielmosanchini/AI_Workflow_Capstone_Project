import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import MultiPolygon
from tqdm import tqdm
import re
import json
import geopandas as gpd
from scripts import ROOT_DIR

# Hong Kong Multipolygon built manually using https://www.keene.edu/campus/maps/tool/
# other geocoords here https://gist.github.com/markmarkoh/2969317
# https://melaniesoek0120.medium.com/data-visualization-how-to-plot-a-map-with-geopandas-in-python-73b10dcd4b4b
RAW_DATA_URL = "https://raw.githubusercontent.com/aavail/ai-workflow-capstone/master/"
DATA_DIR = os.path.join(ROOT_DIR, "data", "datasets")


def download_data():
    """
    Download json input training and production data.

    :return: training and production data.
    """
    data_train_url = os.path.join(RAW_DATA_URL, "cs-train")
    data_production_url = os.path.join(RAW_DATA_URL, "cs-production")

    train_year_month_couples = pd.date_range('2017-11-01', '2019-07-01', freq='MS').strftime("%Y-%m").tolist()
    production_year_month_couples = pd.date_range('2019-08-01', '2019-12-01', freq='MS').strftime("%Y-%m").tolist()

    train_json_data = ["invoices-" + ym + ".json" for ym in train_year_month_couples]
    production_json_data = ["invoices-" + ym + ".json" for ym in production_year_month_couples]

    train_data_urls = [os.path.join(data_train_url, tjd) for tjd in train_json_data]
    production_data_urls = [os.path.join(data_production_url, pjd) for pjd in production_json_data]

    training_data_dfs = []
    for path in tqdm(train_data_urls):
        df = pd.read_json(path)
        training_data_dfs.append(df)

    production_data_dfs = []
    for path in tqdm(production_data_urls):
        df = pd.read_json(path)
        production_data_dfs.append(df)

    return training_data_dfs, production_data_dfs


def validate_downloaded_data(training_data_dfs, production_data_dfs):
    """
    Validate downloaded data, checking for common input errors

    :param training_data_dfs: input dataframes deriving from json training files.
    :param production_data_dfs: input dataframes deriving from json production files.
    """
    # check consistency of number of columns
    col_num = training_data_dfs[0].shape[1]

    for base_data in [training_data_dfs, production_data_dfs]:
        assert all(
            [df.shape[1] == col_num for df in base_data]), "Column number in training/production data is inconsistent."

    # make sure that only customer_id column contains missing values
    general_nan_cols = []
    for base_data in [training_data_dfs, production_data_dfs]:
        for df in base_data:
            nan_cols = df.isna().sum()[df.isna().sum() != 0].index.to_list()
            general_nan_cols.extend(nan_cols)
    assert set(general_nan_cols) == {
        'customer_id'}, "Missing values have been found in columns different than customer_id."


def clean_downloaded_data(training_data_dfs, production_data_dfs):
    """
    Clean downloaded data: correct column names, merge datasets, clean price column,
    clean invoice column, create a date column.

    :param training_data_dfs: input dataframes deriving from json training files.
    :param production_data_dfs: input dataframes deriving from json production files.
    :return: cleaned dataframe with training and production data, ready to be analyzed.
    """
    # correct column names
    column_corrections = {"StreamID": "stream_id",
                          "total_price": "price",
                          "TimesViewed": "times_viewed"}

    training_data_dfs = [bd.rename(columns=column_corrections) for bd in training_data_dfs]
    production_data_dfs = [bd.rename(columns=column_corrections) for bd in production_data_dfs]

    # validating training and production data
    validate_downloaded_data(training_data_dfs, production_data_dfs)

    # merge training_data_dfs and production_data_dfs, respectively
    training_data = pd.concat(training_data_dfs, ignore_index=True)
    production_data = pd.concat(production_data_dfs, ignore_index=True)

    # merge training_data and production_data, adding the column "stage" to account for this
    training_data["stage"] = "training"
    production_data["stage"] = "production"
    merged_data = pd.concat([training_data, production_data], ignore_index=True)

    # remove rows with negative price
    merged_data = merged_data.loc[merged_data["price"] >= 0, :]
    merged_data.reset_index(drop=True, inplace=True)

    # clean invoice column, removing letters and keeping number
    merged_data["invoice"] = merged_data["invoice"].apply(lambda row: re.sub(r"\D", "", row))
    merged_data["invoice"] = merged_data["invoice"].astype("int64")

    # convert year, month, and day to a date column, then drop them
    time_cols = ["year", "month", "day"]
    for col in time_cols:
        merged_data[col] = merged_data[col].astype("str")

    merged_data["date"] = pd.to_datetime(merged_data["month"] + "-" + merged_data["day"] + "-" + merged_data["year"])
    merged_data = merged_data.drop(time_cols, 1)

    return merged_data


def prepare_data(data):
    """
    Prepare data for EDA: download, validate and clean json training and/or production data, keeping only
    the top 10 countries by revenue.

    :param data: what data to return, must be one of 'training', 'production', 'both'.
    :return: training and production data ready to be analyzed.
    """

    assert data in ['training', 'production', 'both'], "data parameter must be one of 'training', 'production', 'both'"

    print("Downloading data ... ")
    try:
        training_data_dfs, production_data_dfs = download_data()
    except Exception as e:
        print("Could not download data")
        raise e
    else:
        print("Successfully downloaded data")

    print("Cleaning data ... ")
    try:
        merged_data = clean_downloaded_data(training_data_dfs, production_data_dfs)
    except Exception as e:
        print("Could not clean data")
        raise e
    else:
        print("Successfully cleaned data")

    if data != "both":
        df = merged_data.loc[merged_data["stage"] == data, :].reset_index(drop=True)
        df.drop("stage", 1, inplace=True)
    else:
        df = merged_data

    # keep only the top 10 countries by revenue
    countries = df.groupby("country")["price"].sum().sort_values(ascending=False).iloc[:10].index.to_list()
    df = df.loc[df["country"].isin(countries), :].reset_index(drop=True)

    return df


def ts_conversion(df, country=None):
    """
    Convert input dataframe to time series version, for a specific country or in general.

    :param df: cleaned input dataframe.
    :param country: if not None, convert dataframe only for a specific country.
    :return: converted dataframe.

    """
    if country is not None:
        country_df = df.loc[df["country"] == country, :].copy()
    else:
        country_df = df.copy()

    grp_df = country_df.groupby(["date"])

    f = {
        'invoice': ['count', 'nunique'],
        'price': ['sum'],
        'stream_id': ['nunique'],
        'times_viewed': ['sum']
    }

    grouped_df = grp_df.agg(f)

    grouped_df.columns = ["purchases", "invoices", "revenue", "streams", "views"]

    grouped_df.index = pd.to_datetime(grouped_df.index)

    return grouped_df


def plot_geodata(df):
    """
    Plot revenue by country on a map, preparing data accordingly.

    :param df: input dataframe.
    """

    country_prices = df.groupby(["country"], as_index=False)["price"].sum()
    country_prices.replace("EIRE", "Ireland", inplace=True)

    # load Hong Kong coords data, built manually
    hk_path = os.path.join(DATA_DIR, "hk_coords.json")
    with open(hk_path, 'r') as f:
        hk = json.load(f)

    # read shapefile using Geopandas
    shapefile_path = os.path.join(DATA_DIR,
                                  'Longitude_Graticules_and_World_Countries_Boundaries-shp',
                                  '99bfd9e7-bb42-4728-87b5-07f8c8ac631c2020328-1-1vef4ev.lu5nk.shp')

    gdf = gpd.read_file(shapefile_path)[['CNTRY_NAME', 'geometry']]
    gdf = gdf.append(pd.DataFrame(data=[["Hong Kong", MultiPolygon(hk)]], columns=gdf.columns), ignore_index=True)

    merged_df = pd.merge(gdf,
                         country_prices,
                         left_on='CNTRY_NAME',
                         right_on='country')

    # bring Hong Kong and Singapore closer to Europe and make them bigger
    trans_scale_values = {"Hong Kong": {"xoff": -85, "yoff": 20, "scale_factor": 15},
                          "Singapore": {"xoff": -85, "yoff": 30, "scale_factor": 15}}

    for country_name in ["Hong Kong", "Singapore"]:
        ts_vals = trans_scale_values[country_name]
        val = merged_df.loc[merged_df["CNTRY_NAME"] == country_name, "geometry"].translate(ts_vals["xoff"],
                                                                                           ts_vals["yoff"]).scale(
            xfact=ts_vals["scale_factor"],
            yfact=ts_vals["scale_factor"])
        merged_df.loc[merged_df["CNTRY_NAME"] == country_name, "geometry"] = val

    # Plot Revenue by country
    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    merged_df.plot(column='price', scheme="quantiles",
                   legend=True,
                   legend_kwds={"loc": "upper left",
                                "title": "revenue intervals"},
                   cmap='coolwarm',
                   ax=ax)

    ax.set_xlim(-13)
    ax.set_ylim(28)
    ax.set_xticklabels([])
    ax.set_xticks([])
    ax.set_yticklabels([])
    ax.set_yticks([])
    ax.set_title("Revenue by country (Top 10)")

    ax.text(x=25, y=46, s="Hong Kong")
    ax.text(x=16, y=34, s="Singapore")

    plt.show()


def plot_revenue_evolution(df):
    """
    Plot interactively a revenue evolution line plot over time for the top 10 countries using Plotly.

    :param df: input dataframe.
    :return: plotly line plot.
    """

    # extract top 10 countries
    countries = df.groupby("country")["price"].sum().sort_values(ascending=False).iloc[:10].index.to_list()

    # extract the revenue evolution for each country
    country_revenue = []
    for cnt_name in countries:
        d = ts_conversion(df, cnt_name)["revenue"]
        country_revenue.append(d.rename(cnt_name))

    pd.options.plotting.backend = "plotly"

    merged_revenue = pd.concat(country_revenue, axis=1)
    merged_revenue = merged_revenue.fillna(0)

    return merged_revenue.plot(title="Revenue evolution in time",
                               labels=dict(index="date", value="revenue", variable="country"))


if __name__ == "__main__":

    # update training and production data downloading them from the web and cleaning them
    merged_df = prepare_data("both")
    train_df = merged_df.loc[merged_df["stage"] == "training", :].reset_index(drop=True)
    prod_df = merged_df.loc[merged_df["stage"] == "production", :].reset_index(drop=True)
    merged_df.drop("stage", 1, inplace=True)

    train_df.to_csv(os.path.join(DATA_DIR, "df_training.csv"), index=False)
    prod_df.to_csv(os.path.join(DATA_DIR, "df_production.csv"), index=False)
