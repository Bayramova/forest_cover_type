![Tests](https://github.com/Bayramova/rs_final_project/actions/workflows/tests.yml/badge.svg)

# Forest Cover Type Prediction
This project trains Extra Tree/Random Forest/Logistic Regression model to classify the forest cover type based on data from corresponding [Kaggle competition](https://www.kaggle.com/competitions/forest-cover-type-prediction).  
[Demo](https://share.streamlit.io/bayramova/forest_cover_type/main/src/main.py) 

## About the Project
The goal of this project is not only to apply machine learning knowledge, but also to learn Python tools and best practices.   
Used tools:
* [Poetry](https://python-poetry.org/docs/) - tool for dependency management and packaging in Python
* [click](https://click.palletsprojects.com/en/8.1.x/) - Python package for creating command line interfaces 
* [pytest](https://docs.pytest.org/en/latest/) - Python testing framework
* [black](https://github.com/psf/black) - Python code formatter
* [flake8](https://github.com/PyCQA/flake8) - Python code linter
* [mypy](https://github.com/python/mypy) - static typing for Python
* [nox](https://nox.thea.codes/en/stable/) - test automation for Python
* [pre-commit](https://pre-commit.com/) - framework for managing and maintaining git hooks
* [GitHub Actions](https://github.com/features/actions) - CI/CD tool

## Usage
1. Make sure Python 3.9 and Poetry are installed on your machine (I use Python 3.9.7 and Poetry 1.1.13).
2. Clone this repository to your machine.
3. Download [Forest train and test datasets](https://www.kaggle.com/competitions/forest-cover-type-prediction) and save csv locally (default paths are *data/train.csv* and *data/test.csv* in repository's root).
4. Install the project dependencies:
```
poetry install --no-dev
```
5. If you want to get more detailed information about the dataset, you can produce EDA report using [Pandas-profiling](https://github.com/ydataai/pandas-profiling) with the following command:
```
poetry run generate-eda -d <path to csv with data> -s <path to save generated EDA report>
```
6. Run train with the following command:
```
poetry run train -d <path to csv with data> -s <path to save trained model>
```
You can configure additional options (such as model, hyperparameters, etc.) in the CLI. To get a full list of them, use help:
```
poetry run train --help
```
7. Run [MLflow UI](https://mlflow.org/docs/latest/tracking.html) to see the information about conducted experiments:
```
poetry run mlflow ui
```
![image_2022-05-01_13-43-38](https://user-images.githubusercontent.com/32398773/166187848-4d19c894-eb57-4f36-8e39-dcd1b1c199f3.png)

8. Run predict to create submission file with predictions:
```
poetry run predict -t <path to csv with test data> -s <path to save submission file> -m <path to the trained model>
```

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

Format your code with [black](https://github.com/psf/black) and lint it with [flake8](https://github.com/PyCQA/flake8):
```
poetry run black src tests noxfile.py
```
```
poetry run flake8 src tests noxfile.py
```
![photo_2022-05-04_11-30-52](https://user-images.githubusercontent.com/32398773/167128312-068f8980-bb2f-4c07-95ac-3ecaa8aad83c.jpg)

Type annotate your code, run [mypy](https://github.com/python/mypy) to ensure the types are correct:
```
poetry run mypy src tests noxfile.py
```
![photo_2022-05-04_15-37-40](https://user-images.githubusercontent.com/32398773/167128571-8417587c-06c8-4cc5-a5c5-e931d9b99a96.jpg)

More conveniently, to run all sessions of testing and formatting in a single command, install and use [nox](https://nox.thea.codes/en/stable/):
```
nox [-r]
```
![photo_2022-05-05_22-09-28](https://user-images.githubusercontent.com/32398773/167128906-7331b9dd-685e-4820-bee4-26dc27a02725.jpg)

If you want to run specific step use the following commands:
```
nox -[r]s lint
nox -[r]s black
nox -[r]s mypy
nox -[r]s tests
```

Install [pre-commit](https://pre-commit.com/) and use it to run automated checks when you commit changes.
Install the [hooks](https://git-scm.com/book/en/v2/Customizing-Git-Git-Hooks) by running the following command:
```
pre-commit install
```
Now pre-commit will run automatically on git commit. To trigger hooks manually for all files use the following command:
```
poetry run pre-commit run --all-files
```
![photo_2022-05-04_12-52-12](https://user-images.githubusercontent.com/32398773/167131234-298687ba-8ca4-4a08-83d8-337dd39580db.jpg)

## Acknowledgments
[Hypermodern Python article series](https://cjolowicz.github.io/posts/hypermodern-python-01-setup/)
