import pandas as pd
import numpy as np

from scripts.supervised_model_utils import resample_df, add_supervised_target, add_supervised_variables, \
    split_train_test, prepare_data_for_model

n = 20
start = "2019-01"
end = "2019-03"
df = pd.DataFrame(data=np.random.random((n, 3)),
                  index=pd.date_range(start, end, periods=n))
df.columns = ["streams", "views", "revenue"]


def test_resample_df():
    new_ind = resample_df(df, "linear").index

    cond = all(new_ind == pd.date_range(start, end))
    assert cond


def test_add_supervised_target():
    resampled_df = resample_df(df, "linear")
    new_df = add_supervised_target(resampled_df, 10)
    target = new_df.loc["2019-01-04", "target"]
    real_target = new_df.loc["2019-01-05":"2019-01-15", "revenue"].sum()

    assert target == real_target


def test_add_supervised_variables():
    resampled_df = resample_df(df, "linear")
    resampled_df.index.name = "date"

    new_df = add_supervised_variables(resampled_df, ("streams", "views"))

    expected_cols = ['streams_3_mean', 'views_3_mean', 'streams_5_mean', 'views_5_mean',
                     'streams_7_mean', 'views_7_mean']

    assert new_df.columns.to_list() == expected_cols


def test_split_train_test():
    sup_df = prepare_data_for_model(None, "train")

    X, y, X_train, y_train, X_test, y_test = split_train_test(sup_df)

    cond_train = X_train.shape[0] + X_test.shape[0] == X.shape[0]
    cond_test = y_train.shape[0] + y_test.shape[0] == y.shape[0]

    assert cond_train and cond_test
