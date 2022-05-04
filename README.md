Final project for RS School Machine Learning course.

This demo uses [Forest Cover Type](https://www.kaggle.com/competitions/forest-cover-type-prediction) dataset.

## Usage
This package allows you to train model to predict the forest cover type (the predominant kind of tree cover) from cartographic variables.
1. Clone this repository to your machine.
2. Download [Forest train dataset](https://www.kaggle.com/competitions/forest-cover-type-prediction), save csv locally (default path is *data/train.csv* in repository's root).
3. Make sure Python 3.9 and Poetry are installed on your machine (I use Python 3.9.7 and Poetry 1.1.13).
4. Install the project dependencies:
```
poetry install --no-dev
```
5. If you want to get more detailed information about the dataset, you can generate an EDA report with the following command:
```
poetry run generate-eda -d <path to csv with data> -s <path to save generated EDA report>
```
6. Run train with the following command:
```
poetry run train -d <path to csv with data> -s <path to save trained model>
```
You can configure additional options (such as model, hyperparameters tuning configuration, etc.) in the CLI. To get a full list of them, use help:
```
poetry run train --help
```
7. Run MLflow UI to see the information about experiments you conducted:
```
poetry run mlflow ui
```
![image_2022-05-01_13-43-38](https://user-images.githubusercontent.com/32398773/166187848-4d19c894-eb57-4f36-8e39-dcd1b1c199f3.png)

**Note**: to produce the same results as mine, don't specify random seeds manually (the default random seed is 42). The results may still differ from what you see in the screenshot, because I made it before applying automatic hyperparameter search and nested CV. By the way, hyperparameter tuning takes some time, so be patient.
