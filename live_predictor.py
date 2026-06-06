import fastf1
import pandas as pd
import numpy as np
import joblib

from features_utils import get_driver_features

model = joblib.load("f1_model.pkl")

YEAR = 2026
TRACK = "Monaco"

fp1 = fastf1.get_session(YEAR, TRACK, "FP1")

fp2 = fastf1.get_session(YEAR, TRACK, "FP2")

quali = fastf1.get_session(YEAR, TRACK, "Q")

fp1.load(weather=True)
fp2.load(weather=True)
quali.load()

weather = quali.weather_data

rows = []

for d in fp1.results["Abbreviation"]:

    feats = get_driver_features(fp1=fp1, fp2=fp2, quali=quali, weather=weather, driver=d, year=YEAR)
    rows.append(feats)

live_df = pd.DataFrame(rows)

live_df = pd.get_dummies(live_df, columns=["team"])

live_df = live_df.reindex(columns=model.feature_names_in_, fill_value=0)

pred = model.predict(live_df)

live_df["score"] = pred

# creating a proper ranking race outcome
live_df["predicted_finish"] = (live_df["score"].rank(method="first").astype(int))

print(live_df.sort_values("predicted_finish")[["driver", "predicted_finish"]])
