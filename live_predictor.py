import fastf1
import pandas as pd
import numpy as np
import joblib

fastf1.Cache.enable_cache("f1_cache")

model = joblib.load("f1_model.pkl")

def get_pace(session, driver):
    """
    Returns race pace of the driver in the session by computing median lap time in session
    """

    laps = session.laps.pick_driver(driver)

    # picking only quick laps to avoid the outlaps
    laps = laps.pick_quicklaps()

    if laps.empty:
        return np.nan

    return laps['LapTime'].dt.total_seconds().median()

def get_driver_features(fp1, fp2, quali, race, driver, weather, year):

    race_results = race.results.set_index("Abbreviation")
    quali_results = quali.results.set_index("Abbreviation")

    if driver not in race_results.index or driver not in quali_results.index:
        return None

    r_driver = race_results.loc[driver]
    q_driver = quali_results.loc[driver]

    fp1_pace = get_pace(fp1, driver)
    fp2_pace = get_pace(fp2, driver)

    return {
        "year": year,
        "driver": driver,

        "fp1_pace": fp1_pace,
        "fp2_pace": fp2_pace,
        "pace_dif": fp1_pace - fp2_pace if pd.notnull(fp1_pace) and pd.notnull(fp2_pace) else np.nan,

        "quali_pos": q_driver['Position'],
        "grid_pos": r_driver['GridPosition'],

        "team": r_driver['TeamName'],

        "air_temp": weather["AirTemp"].mean(),
        "track_temp": weather["TrackTemp"].mean(),
        "rainfall": weather["Rainfall"].max(),
    }


YEAR = 2026
TRACK = 'Monaco'
fp1 = fastf1.get_session(YEAR, TRACK, "FP1")
fp2 = fastf1.get_session(YEAR, TRACK, "FP2")
quali = fastf1.get_session(YEAR, TRACK, "Q")
race = fastf1.get_session(YEAR, TRACK, "R")

fp1.load()
fp2.load()
quali.load()
race.load()

weather = race.weather_data

rows = []

drivers = quali.results["Abbreviation"].unique()

for d in drivers:
    feats = get_driver_features(fp1, fp2, quali, race, d, weather, YEAR)

    if feats is not None:
        rows.append(feats)

live_df = pd.DataFrame(rows)

live_df = pd.get_dummies(live_df, columns=["driver", "team"])

model_features = model.feature_names_in_

live_df = live_df.reindex(columns=model_features, fill_value=0)

live_df["predicted_finish"] = model.predict(live_df)

live_df = live_df.sort_values("predicted_finish")

live_df["win_score"] = 1 / (live_df["predicted_finish"] + 1e-6)

print("\n🏁 LIVE MONACO PREDICTION 🏁\n")

print(live_df[[
    "predicted_finish",
    "win_score"
]].head(10))
