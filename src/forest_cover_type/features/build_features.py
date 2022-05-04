import numpy as np


def build_features(X, bin_elevation, log_transform):
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
    return data
