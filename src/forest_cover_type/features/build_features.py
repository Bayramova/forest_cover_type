import click
import numpy as np
import pandas as pd


def build_features(
    X: pd.DataFrame, bin_elevation: bool, log_transform: bool
) -> pd.DataFrame:
    data = X.copy()
    if bin_elevation:
        data["Binned_Elevation"] = data["Elevation"].apply(lambda x: np.floor(x / 50))
    if log_transform:
        data["Horizontal_Distance_To_Roadways_Log"] = data[
            "Horizontal_Distance_To_Roadways"
        ].apply(lambda x: np.log(x, where=(x != 0)))
        # data["Horizontal_Distance_To_Roadways_Log"] = data[
        #     "Horizontal_Distance_To_Roadways"
        # ].apply(lambda x: np.log(x + 1))
        data["Horizontal_Distance_To_Fire_Points_Log"] = data[
            "Horizontal_Distance_To_Fire_Points"
        ].apply(lambda x: np.log(x, where=(x != 0)))
        # data["Horizontal_Distance_To_Fire_Points_Log"] = data[
        #     "Horizontal_Distance_To_Fire_Points"
        # ].apply(lambda x: np.log(x + 1))
        data["Horizontal_Distance_To_Hydrology_Log"] = data[
            "Horizontal_Distance_To_Hydrology"
        ].apply(lambda x: np.log(x, where=(x != 0)))
        # data["Horizontal_Distance_To_Hydrology_Log"] = data[
        #     "Horizontal_Distance_To_Hydrology"
        # ].apply(lambda x: np.log(x + 1))
    click.echo(f"Dataset shape after feature engineering: {data.shape}")
    return data
