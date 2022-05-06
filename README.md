![Tests](https://github.com/Bayramova/rs_final_project/actions/workflows/tests.yml/badge.svg)

Final project for RS School Machine Learning course.

This project uses [Forest Cover Type](https://www.kaggle.com/competitions/forest-cover-type-prediction) dataset.

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

## Development

The code in this repository must be tested, linted with flake8, formatted with black, and pass mypy typechecking before being commited to the repository.

Install all requirements (including dev requirements) to poetry environment:
```
poetry install
```
Now you can use developer instruments, e.g. pytest:
```
poetry run pytest
```
![image_2022-05-03_19-56-42](https://user-images.githubusercontent.com/32398773/167127639-e7854ee2-5141-4ff9-8d90-0b82b50ad989.png)

More conveniently, to run all sessions of testing and formatting in a single command, install and use [nox](https://nox.thea.codes/en/stable/):
```
nox [-r]
```
![photo_2022-05-05_22-09-28](https://user-images.githubusercontent.com/32398773/167128906-7331b9dd-685e-4820-bee4-26dc27a02725.jpg)

Format your code with [black](https://github.com/psf/black) and lint it with [flake8](https://github.com/PyCQA/flake8) by using either nox or poetry:
```
nox -[r]s black
```
```
poetry run black src tests noxfile.py
```
```
nox -[r]s lint
```
```
poetry run flake8
```
![photo_2022-05-04_11-30-52](https://user-images.githubusercontent.com/32398773/167128312-068f8980-bb2f-4c07-95ac-3ecaa8aad83c.jpg)

Type annotate your code, run mypy to ensure the types are correct by using either nox or poetry:
```
nox -[r]s mypy
```
```
poetry run mypy src tests noxfile.py
```
![photo_2022-05-04_15-37-40](https://user-images.githubusercontent.com/32398773/167128571-8417587c-06c8-4cc5-a5c5-e931d9b99a96.jpg)

Install [pre-commit](https://pre-commit.com/) and use it to run automated checks when you commit changes.
Install the [hooks](https://git-scm.com/book/en/v2/Customizing-Git-Git-Hooks) by running the following command:
```
pre-commit install
```
The hooks run automatically every time you invoke git commit. To trigger them manually for all files use the following command:
```
pre-commit run --all-files
```
![photo_2022-05-04_12-52-12](https://user-images.githubusercontent.com/32398773/167131234-298687ba-8ca4-4a08-83d8-337dd39580db.jpg)
