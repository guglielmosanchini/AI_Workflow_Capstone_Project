import argparse
from flask import Flask, jsonify, request, render_template, send_from_directory
import os
import re
import numpy as np
import pandas as pd

from scripts.supervised_model_utils import load_model, score_model, train_model
from scripts import ROOT_DIR, AVAILABLE_COUNTRIES

LOG_DIR = os.path.join(ROOT_DIR, "logs")

app = Flask(__name__)


@app.route("/")
def landing():
    return "Hi there!"


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """
    basic predict function for the API
    """

    # input checking
    if not request.json:
        print("ERROR: API (predict): did not receive request data.")
        return jsonify([])

    if 'query' not in request.json:
        print("ERROR API (predict): received request, but no 'query' found within.")
        return jsonify([])

    if not isinstance(request.json["query"], list):
        print("ERROR API (predict): query must be a list containing a dictionary having 'country'"
              " and 'starting_dates' as keys.")
        return jsonify([])

    if not isinstance(request.json["query"][0], dict):
        print("ERROR API (predict): query must be a list containing a dictionary having 'country'"
              " and 'starting_dates' as keys.")
        return jsonify([])

    if "country" not in request.json["query"][0]:
        print("ERROR API (predict): missing 'country' specification.")
        return jsonify([])

    if request.json["query"][0]["country"] not in AVAILABLE_COUNTRIES:
        print("ERROR API (predict): acceptable countries are", *AVAILABLE_COUNTRIES)
        return jsonify([])

    if "starting_dates" not in request.json["query"][0]:
        print("ERROR API (predict): missing 'starting_dates' specification.")
        return jsonify([])

    try:
        _ = [pd.Timestamp(sd) for sd in request.json["query"][0]["starting_dates"]]
    except ValueError:
        print("ERROR API (predict): 'starting_dates' cannot be parsed correctly as dates.")
        return jsonify([])

    # extract the query
    query = request.json['query'][0]

    try:
        model, model_name = load_model(country_name=query['country'])
    except FileNotFoundError:
        print("ERROR: no model is available for this country. Train a model and retry.")
        return jsonify([])

    _result = score_model(query['starting_dates'], model_name)
    result = []

    # convert numpy objects to ensure they are serializable
    for item in _result:
        if isinstance(item, np.ndarray):
            result.append(item.tolist())
        else:
            result.append(item)

    return jsonify(result)


@app.route('/train', methods=['GET', 'POST'])
def train():

    # check for request data
    if not request.json:
        print("ERROR: API (train): did not receive request data.")
        return jsonify(False)

    if 'query' not in request.json:
        print("ERROR API (train): received request, but no 'query' found within.")
        return jsonify([])

    if 'country' not in request.json['query'][0]:
        print("ERROR API (train): missing 'country' field in query.")
        return jsonify([])

    if request.json["query"][0]["country"] not in AVAILABLE_COUNTRIES:
        print("ERROR API (predict): acceptable countries are", *AVAILABLE_COUNTRIES)
        return jsonify([])

    query = request.json['query'][0]

    test = query.get('test')
    if test is None:
        test = False

    model = train_model(query['country'], query.get('param_dim'), test)
    print("... training complete")

    return jsonify(True)


@app.route('/logs/<filename>', methods=['GET'])
def logs(filename):
    """
    API endpoint to get logs
    """
    if not re.search(".log", filename):
        print("ERROR: API (log): file requested was not a log file: {}.".format(filename))
        return jsonify([])

    if not os.path.isdir(LOG_DIR):
        print("ERROR: API (log): cannot find log dir.")
        return jsonify([])

    file_path = os.path.join(LOG_DIR, filename)
    if not os.path.exists(file_path):
        print("ERROR: API (log): file requested could not be found: {}.".format(filename))
        return jsonify([])

    return send_from_directory(LOG_DIR, filename, as_attachment=True)


if __name__ == '__main__':

    # parse arguments for debug mode
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--debug", action="store_true", help="debug flask")
    args = vars(ap.parse_args())

    if args["debug"]:
        app.run(debug=True, port=8080)
    else:
        app.run(host='0.0.0.0', threaded=True, port=8080)

