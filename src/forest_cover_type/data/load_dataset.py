import pandas as pd
from sklearn.model_selection import train_test_split


def load_dataset(dataset_path, test_split_ratio, random_state):
    data = pd.read_csv(dataset_path, index_col="Id")
    X = data.drop(columns=['Cover_Type'])
    y = data['Cover_Type']
    return train_test_split(
        X, y, test_size=test_split_ratio, random_state=random_state)
