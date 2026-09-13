from fastapi import FastAPI
import fastf1
import pandas as pd
import numpy as np
import joblib
from fastapi.middleware.cors import CORSMiddleware
from features_utils import get_driver_features, get_historical_features, build_history
from database import get_race_data

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

YEAR = 2026

fastf1.Cache.enable_cache(".fastf1_cache")

model = joblib.load("f1_model.pkl")
model_features = joblib.load("model_features.pkl")
historical_df = get_race_data()
history = build_history(historical_df)


@app.get("/")
def home():
    return {"message": "F1 Prediction API is running"}


@app.get("/predict")
def predict(track: str):
    fp1 = fastf1.get_session(YEAR, track, "FP1")
    fp2 = fastf1.get_session(YEAR, track, "FP2")
    quali = fastf1.get_session(YEAR, track, "Q")

    fp1.load()
    fp2.load()
    quali.load()

    try:
        weather = quali.weather_data
    except Exception:
        weather = pd.DataFrame({"AirTemp": [np.nan], "TrackTemp": [np.nan], "Rainfall": [0]})

    rows = []

    for driver in fp1.results["Abbreviation"]:
        feats = get_driver_features(fp1, fp2, weather, driver, YEAR, track, quali)

        if feats is None:
            continue

        team = feats["team"]
        historical = get_historical_features(history, driver, team, track)
        feats.update(historical)
        rows.append(feats)

    live_df = pd.DataFrame(rows)

    if live_df.empty:
        return {"error": "No usable driver data found."}

    drivers = live_df["driver"].copy()

    live_df = pd.get_dummies(live_df, columns=["driver", "team", "track"])
    live_df = live_df.reindex(columns=model_features, fill_value=0)

    predictions = model.predict(live_df)

    results = pd.DataFrame({"driver": drivers, "predicted_finish": predictions})
    results["predicted_position"] = results["predicted_finish"].rank(method="first").astype(int)
    results = results.sort_values("predicted_position")

    return {
        "year": YEAR,
        "track": track,
        "predictions": results.to_dict("records")
    }