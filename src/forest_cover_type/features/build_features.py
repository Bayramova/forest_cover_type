import click
import numpy as np
import pandas as pd


def build_features(X: pd.DataFrame) -> pd.DataFrame:
    data = X.copy()

    data["Horizontal_Distance_To_Roadways_Log"] = data[
        "Horizontal_Distance_To_Roadways"
    ].apply(lambda x: np.log(x + 1))
    data["Horizontal_Distance_To_Fire_Points_Log"] = data[
        "Horizontal_Distance_To_Fire_Points"
    ].apply(lambda x: np.log(x + 1))

    data["Hydro_Fire_1"] = (
        data["Horizontal_Distance_To_Hydrology"]
        + data["Horizontal_Distance_To_Fire_Points"]
    )
    data["Hydro_Fire_2"] = (
        data["Horizontal_Distance_To_Hydrology"]
        - data["Horizontal_Distance_To_Fire_Points"]
    )
    data["Mean_Distance_Hydrology_Roadways"] = (
        data["Horizontal_Distance_To_Hydrology"]
        + data["Horizontal_Distance_To_Roadways"]
    ) / 2
    data["Hydro_Road_1"] = (
        data["Horizontal_Distance_To_Hydrology"]
        + data["Horizontal_Distance_To_Roadways"]
    )
    data["Hydro_Road_2"] = (
        data["Horizontal_Distance_To_Hydrology"]
        - data["Horizontal_Distance_To_Roadways"]
    )
    data["Mean_Distance_Firepoints_Roadways"] = (
        data["Horizontal_Distance_To_Fire_Points"]
        + data["Horizontal_Distance_To_Roadways"]
    ) / 2
    data["Fire_Road_1"] = (
        data["Horizontal_Distance_To_Fire_Points"]
        + data["Horizontal_Distance_To_Roadways"]
    )
    data["Fire_Road_2"] = (
        data["Horizontal_Distance_To_Fire_Points"]
        - data["Horizontal_Distance_To_Roadways"]
    )
    data["Elev_Hydro_1"] = abs(
        data["Elevation"] - data["Vertical_Distance_To_Hydrology"]
    )
    data["Elev_Hydro_2"] = data["Elevation"] - data["Vertical_Distance_To_Hydrology"]
    data["Elev_Hydro_3"] = (
        data["Elevation"] - data["Horizontal_Distance_To_Hydrology"] * 0.2
    )
    data["Elev_Road_1"] = data["Elevation"] + data["Horizontal_Distance_To_Roadways"]
    data["Elev_Road_2"] = data["Elevation"] - data["Horizontal_Distance_To_Roadways"]
    data["Elev_Fire_1"] = data["Elevation"] + data["Horizontal_Distance_To_Fire_Points"]
    data["Elev_Fire_2"] = data["Elevation"] - data["Horizontal_Distance_To_Fire_Points"]

    data["Soil_Type"] = data.iloc[
        :, data.columns.get_loc("Soil_Type1") : data.columns.get_loc("Soil_Type40") + 1
    ].idxmax(axis=1)
    data["Soil_Type"] = data["Soil_Type"].apply(
        lambda x: int(x[-1]) if len(x) == 10 else int(x[-2:])
    )

    climatic_zone = {
        1: 2,
        2: 2,
        3: 2,
        4: 2,
        5: 2,
        6: 2,
        7: 3,
        8: 3,
        9: 4,
        10: 4,
        11: 4,
        12: 4,
        13: 4,
        14: 5,
        15: 5,
        16: 6,
        17: 6,
        18: 6,
        19: 7,
        20: 7,
        21: 7,
        22: 7,
        23: 7,
        24: 7,
        25: 7,
        26: 7,
        27: 7,
        28: 7,
        29: 7,
        30: 7,
        31: 7,
        32: 7,
        33: 7,
        34: 7,
        35: 8,
        36: 8,
        37: 8,
        38: 8,
        39: 8,
        40: 8,
    }
    geologic_zone = {
        1: 7,
        2: 7,
        3: 7,
        4: 7,
        5: 7,
        6: 7,
        7: 5,
        8: 5,
        9: 2,
        10: 7,
        11: 7,
        12: 7,
        13: 7,
        14: 1,
        15: 1,
        16: 1,
        17: 1,
        18: 7,
        19: 1,
        20: 1,
        21: 1,
        22: 2,
        23: 2,
        24: 7,
        25: 7,
        26: 7,
        27: 7,
        28: 7,
        29: 7,
        30: 7,
        31: 7,
        32: 7,
        33: 7,
        34: 7,
        35: 7,
        36: 7,
        37: 7,
        38: 7,
        39: 7,
        40: 7,
    }
    surface_cover = {
        1: 3,
        2: 2,
        3: 4,
        4: 4,
        5: 4,
        6: 1,
        7: 0,
        8: 0,
        9: 2,
        10: 4,
        11: 4,
        12: 1,
        13: 4,
        14: 0,
        15: 0,
        16: 0,
        17: 0,
        18: 2,
        19: 0,
        20: 0,
        21: 0,
        22: 3,
        23: 0,
        24: 3,
        25: 3,
        26: 2,
        27: 3,
        28: 3,
        29: 3,
        30: 3,
        31: 3,
        32: 3,
        33: 3,
        34: 3,
        35: 0,
        36: 3,
        37: 3,
        38: 3,
        39: 3,
        40: 3,
    }
    rock_size = {
        1: 1,
        2: 1,
        3: 3,
        4: 3,
        5: 3,
        6: 1,
        7: 0,
        8: 0,
        9: 1,
        10: 3,
        11: 3,
        12: 1,
        13: 3,
        14: 0,
        15: 0,
        16: 0,
        17: 0,
        18: 1,
        19: 0,
        20: 0,
        21: 0,
        22: 2,
        23: 0,
        24: 1,
        25: 1,
        26: 1,
        27: 1,
        28: 1,
        29: 1,
        30: 1,
        31: 1,
        32: 1,
        33: 1,
        34: 1,
        35: 0,
        36: 1,
        37: 1,
        38: 1,
        39: 1,
        40: 1,
    }
    data["Climatic_Zone"] = data["Soil_Type"].map(climatic_zone)
    data["Geologic_Zone"] = data["Soil_Type"].map(geologic_zone)
    data["Surface_Cover"] = data["Soil_Type"].map(surface_cover)
    data["Rock_Size"] = data["Soil_Type"].map(rock_size)

    # Important Soil Types
    data["Soil_12_32"] = data["Soil_Type32"] + data["Soil_Type12"] + data["Soil_Type29"]
    data["Soil_Type23_22_32_33"] = (
        data["Soil_Type23"]
        + data["Soil_Type22"]
        + data["Soil_Type32"]
        + data["Soil_Type33"]
    )

    # Soil Type Interactions
    data["Soil29_Area1"] = data["Soil_Type29"] + data["Wilderness_Area1"]
    data["Soil30_Area1"] = data["Soil_Type30"] + data["Wilderness_Area1"]
    data["Soil27_Area1"] = data["Soil_Type27"] + data["Wilderness_Area1"]

    data["Soil3_Area4"] = data["Wilderness_Area4"] + data["Soil_Type3"]

    #  New Features Interactions
    data["Climate_Area2"] = data["Wilderness_Area2"] * data["Climatic_Zone"]
    data["Climate_Area4"] = data["Wilderness_Area4"] * data["Climatic_Zone"]

    data["Rock_Area1"] = data["Wilderness_Area1"] * data["Rock_Size"]
    data["Rock_Area3"] = data["Wilderness_Area3"] * data["Rock_Size"]

    data["Surface_Area1"] = data["Wilderness_Area1"] * data["Surface_Cover"]
    data["Surface_Area2"] = data["Wilderness_Area2"] * data["Surface_Cover"]
    data["Surface_Area4"] = data["Wilderness_Area4"] * data["Surface_Cover"]

    # Drop original soil features
    soil_features = [f"Soil_Type{i}" for i in range(1, 41)]
    data.drop(columns=soil_features, inplace=True)

    cols = data.columns
    data = data[cols]
    click.echo(f"Dataset shape after feature engineering: {data.shape}")
    return data
