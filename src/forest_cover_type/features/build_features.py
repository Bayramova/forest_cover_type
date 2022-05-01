import numpy as np


def build_features(X, bin_elevation):
    data = X.copy()
    if bin_elevation:
        data["Binned_Elevation"] = data["Elevation"].apply(
            lambda x: np.floor(x/50))
    return data
