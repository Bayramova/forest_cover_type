from typing import Any, List

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from forest_cover_type.features.build_features import build_features

model_path = "models/model.joblib"
columns = [
    "Elevation",
    "Aspect",
    "Slope",
    "Horizontal_Distance_To_Hydrology",
    "Vertical_Distance_To_Hydrology",
    "Horizontal_Distance_To_Roadways",
    "Hillshade_9am",
    "Hillshade_Noon",
    "Hillshade_3pm",
    "Horizontal_Distance_To_Fire_Points",
    "Wilderness_Area",
    "Soil_Type",
    "Wilderness_Area1",
    "Wilderness_Area2",
    "Wilderness_Area3",
    "Wilderness_Area4",
    "Soil_Type1",
    "Soil_Type2",
    "Soil_Type3",
    "Soil_Type4",
    "Soil_Type5",
    "Soil_Type6",
    "Soil_Type7",
    "Soil_Type8",
    "Soil_Type9",
    "Soil_Type10",
    "Soil_Type11",
    "Soil_Type12",
    "Soil_Type13",
    "Soil_Type14",
    "Soil_Type15",
    "Soil_Type16",
    "Soil_Type17",
    "Soil_Type18",
    "Soil_Type19",
    "Soil_Type20",
    "Soil_Type21",
    "Soil_Type22",
    "Soil_Type23",
    "Soil_Type24",
    "Soil_Type25",
    "Soil_Type26",
    "Soil_Type27",
    "Soil_Type28",
    "Soil_Type29",
    "Soil_Type30",
    "Soil_Type31",
    "Soil_Type32",
    "Soil_Type33",
    "Soil_Type34",
    "Soil_Type35",
    "Soil_Type36",
    "Soil_Type37",
    "Soil_Type38",
    "Soil_Type39",
    "Soil_Type40",
]
cover_types = {
    1: "Spruce/Fir",
    2: "Lodgepole Pine",
    3: "Ponderosa Pine",
    4: "Cottonwood/Willow",
    5: "Aspen",
    6: "Douglas-fir",
    7: "Krummholz",
}


def predict(input: List[Any]) -> str:
    input_df = pd.DataFrame(
        np.hstack((np.array(input), np.zeros(44))).reshape(1, -1), columns=columns
    )
    w_area = int(input_df["Wilderness_Area"].values[0])
    soil_type = int(input_df["Soil_Type"].values[0])
    input_df[f"Wilderness_Area{w_area}"] = 1
    input_df[f"Soil_Type{soil_type}"] = 1
    input_df = input_df.drop(["Wilderness_Area"], axis=1)
    input_df = input_df.drop(["Soil_Type"], axis=1)

    X = build_features(input_df)

    model = joblib.load(model_path)

    prediction = model.predict(X)[0]

    return f"Cover type is {prediction}: {cover_types[prediction]}"


# print(predict([2693, 96, 30, 180, 3, 2672, 251, 183, 38, 6635, 1, 30]))


st.title("Forest Cover Type Prediction")

# get the input data from the user
Elevation = st.number_input("Elevation in meters", 0)
Aspect = st.number_input("Aspect in degrees azimuth", 0, 360)
Slope = st.number_input("Slope in degrees", 0, 90)
Horizontal_Distance_To_Hydrology = st.number_input(
    "Horz Dist to nearest surface water features", 0
)
Vertical_Distance_To_Hydrology = st.number_input(
    "Vert Dist to nearest surface water features"
)
Horizontal_Distance_To_Roadways = st.number_input("Horz Dist to nearest roadway", 0)
Hillshade_9am = st.number_input("Hillshade index at 9am, summer solstice", 0, 255)
Hillshade_Noon = st.number_input("Hillshade index at noon, summer solstice", 0, 255)
Hillshade_3pm = st.number_input("Hillshade index at 3pm, summer solstice", 0, 255)
Horizontal_Distance_To_Fire_Points = st.number_input(
    "Horz Dist to nearest wildfire ignition points", 0
)
Wilderness_Area = st.number_input("Wilderness area designation", 1, 4)
Soil_Type = st.number_input("Soil Type designation", 1, 40)

# predict
prediction = ""
if st.button("Predict"):
    prediction = predict(
        [
            Elevation,
            Aspect,
            Slope,
            Horizontal_Distance_To_Hydrology,
            Vertical_Distance_To_Hydrology,
            Horizontal_Distance_To_Roadways,
            Hillshade_9am,
            Hillshade_Noon,
            Hillshade_3pm,
            Horizontal_Distance_To_Fire_Points,
            Wilderness_Area,
            Soil_Type,
        ]
    )

    st.success(prediction)
