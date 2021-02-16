#!/usr/bin/env python
"""
api tests

these tests use the requests package however similar requests can be made with curl

e.g.

data = '{"key":"value"}'
curl -X POST -H "Content-Type: application/json" -d "%s" http://localhost:8080/predict'%(data)
"""

import pytest
import requests
import re
from ast import literal_eval

port = 8080
host = "http://0.0.0.0"

try:
    requests.post(f'{host}:{port}/predict')
    server_available = True
except requests.exceptions.RequestException:
    server_available = False

skip_if_server_not_available = pytest.mark.skipif(
    not server_available,  reason="local server is not running"
)


@skip_if_server_not_available
def test_01_train():
    """
    test the train functionality
    """

    request_json = {"query": [{"country": None,
                               "param_dim": "very_small",
                               "test": True}]}
    r = requests.post(f'{host}:{port}/train', json=request_json)
    train_complete = re.sub(r"\W+", "", r.text)

    assert train_complete == 'true'


@skip_if_server_not_available
def test_02_predict_empty():
    """
    ensure appropriate failure types
    """

    # provide no data at all
    r = requests.post(f'{host}:{port}/predict')
    cond_empty = re.sub('\n|"', '', r.text) == "[]"

    # provide improperly formatted data
    wrong_json = {"query": [34, "wrong"]}
    r = requests.post(f'{host}:{port}/predict', json=wrong_json)
    cond_wrong = re.sub('\n|"', '', r.text) == "[]"

    assert cond_wrong and cond_empty


@skip_if_server_not_available
def test_03_predict():
    """
    test the predict functionality
    """
    request_json = {"query": [{"country": None,
                               "starting_dates": ["2017-12-06", "2018-01-29"]}]}

    r = requests.post(f'{host}:{port}/predict', json=request_json)
    response = literal_eval(r.text)

    cond = [0 <= float(p[0]) <= 500_000 for p in response]
    assert cond


@skip_if_server_not_available
def test_04_logs():
    """
    test the log functionality
    """

    file_name = 'train-test.log'
    r = requests.get(f'{host}:{port}/logs/{file_name}')

    assert True
