import fastf1
import pandas as pd
import numpy as np
import joblib
from database import get_race_data
from features_utils import get_driver_features, get_historical_features, build_history


YEAR = 2026
TRACK = input("Enter the race you want to predict: ")

fastf1.Cache.enable_cache(".fastf1_cache")

model = joblib.load("f1_model.pkl")
model_features = joblib.load("model_features.pkl")
historical_df = get_race_data()

history = build_history(historical_df)

print(f"\nLoading {YEAR} {TRACK}...")

fp1 = fastf1.get_session(YEAR, TRACK, "FP1")
fp2 = fastf1.get_session(YEAR, TRACK, "FP2")
quali = fastf1.get_session(YEAR, TRACK, "Q")

fp1.load()
fp2.load()
quali.load()

try:
    weather = quali.weather_data
except Exception:
    weather = pd.DataFrame({"AirTemp": [np.nan], "TrackTemp": [np.nan], "Rainfall": [0]})

rows = []

for driver in fp1.results["Abbreviation"]:
    feats = get_driver_features(fp1, fp2, weather, driver, YEAR, TRACK, quali)

    if feats is None:
        continue

    team = feats["team"]
    historical = get_historical_features(history, driver, team, TRACK)
    feats.update(historical)
    rows.append(feats)

live_df = pd.DataFrame(rows)

if live_df.empty:
    print("\nNo usable driver data found.")
    exit()

drivers = live_df["driver"].copy()

live_df = pd.get_dummies(live_df, columns=["driver", "team", "track"])
live_df = live_df.reindex(columns=model_features, fill_value=0)

predictions = model.predict(live_df)

results = pd.DataFrame({"driver": drivers, "predicted_finish": predictions})
results["predicted_position"] = results["predicted_finish"].rank(method="first").astype(int)
results = results.sort_values("predicted_position")

print("\n==============================")
print(f"Predicted {TRACK} {YEAR} Results")
print("==============================")
print(results[["predicted_position", "driver", "predicted_finish"]].to_string(index=False))