import time
import os
import csv
import uuid
from datetime import date
from scripts import ROOT_DIR

LOG_DIR = os.path.join(ROOT_DIR, "logs")

if not os.path.exists(LOG_DIR):
    os.mkdir(LOG_DIR)


def update_train_log(model_params, eval_test, param_dim, runtime, MODEL_VERSION, test=False):
    """
    update train log file
    """
    today = date.today()

    if test is False:
        log_name = "train-{}-{}-{}.log".format(today.year,
                                               today.month,
                                               today.day)
    else:
        log_name = "train-test.log"

    logfile = os.path.join(LOG_DIR, log_name)
    # write the data to a csv file
    header = ['unique_id', 'timestamp', 'model_params', 'eval_test', 'param_dim', 'model_version', 'runtime']
    write_header = False
    if not os.path.exists(logfile):
        write_header = True
    with open(logfile, 'a') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        if write_header:
            writer.writerow(header)

        to_write = map(str, [uuid.uuid4(), time.time(), model_params, eval_test, param_dim,
                             MODEL_VERSION, runtime])
        writer.writerow(to_write)


def update_predict_log(y_pred, query, runtime, MODEL_VERSION, test=False):
    """
    update predict log file
    """
    today = date.today()

    if test is False:
        log_name = "predict-{}-{}-{}.log".format(today.year,
                                               today.month,
                                               today.day)
    else:
        log_name = "predict-test.log"

    logfile = os.path.join(LOG_DIR, log_name)
    # write the data to a csv file
    header = ['unique_id', 'timestamp', 'y_pred', 'query', 'model_version', 'runtime']
    write_header = False
    if not os.path.exists(logfile):
        write_header = True
    with open(logfile, 'a') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        if write_header:
            writer.writerow(header)

        to_write = map(str, [uuid.uuid4(), time.time(), y_pred, query, MODEL_VERSION, runtime])
        writer.writerow(to_write)
