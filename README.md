# IBM AI Workflow Capstone Project

https://github.com/aavail/ai-workflow-capstone

## Tests
To run all tests at once:
```python 
python -m pytest 
```
To run specific tests:
```python 
python -m pytest tests/{filename}
```
where `{filename}` can be `test_api.py`, `test_logger.py` or `test_model.py`.

## Scripts

The `scripts` package contains modules to download and clean data, to transform data and build the model(s), to build 
loggers and to monitor the model in production.

## Data

The `data` folder contains:
- some binary files used to plot;
- the folder where models and model configs are saved;
- the folder where training and production datasets are saved.

## Docker
To build a Docker image:
```bash
docker build -t image_name .
```
To see Docker images:
```bash
docker images
```
To run it:
```bash
docker run -p 4000:8080 image_name
```
then go to http://0.0.0.0:4000/.

## Update training and production data

To update training and production data, run:
```python 
export PYTHONPATH="."
python scripts/utils.py
```
The new training and production dataframes will be saved in `./data/datasets`.

## Monitor production performance
```python 
export PYTHONPATH="."
python scripts/monitoring.py
```
The score of the model on production data will be printed, and some plots shown.


## Additional info

What could have been done more:

- perform extensive gridsearch for country-specific models
- add confidence intervals estimation to provide a risk metric.
- use a non supervised approachs.
- add more tests.
- take security into account, e.g. using The Adversarial Robustness Toolbox.
- add periodic automatic retraining.