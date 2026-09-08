import fastf1
import pandas as pd
import numpy as np
import joblib
from features_utils import get_driver_features

user_race = input("Enter the upcoming race you want to predict: ")

YEAR = 2026
TRACK = user_race

model = joblib.load(f"{TRACK}_model.pkl")

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
live_df = live_df.fillna(0)

drivers = live_df["driver"]

live_df = pd.get_dummies(live_df, columns=["driver", "team"])

live_df = live_df.reindex(columns=model.feature_names_in_, fill_value=0)

pred = model.predict(live_df)

live_df["driver"] = drivers

live_df["score"] = pred

# creating a proper ranking race outcome
live_df["predicted_finish"] = (live_df["score"].rank(method="first").astype(int))

print(live_df.sort_values("predicted_finish")[["driver", "predicted_finish"]])
