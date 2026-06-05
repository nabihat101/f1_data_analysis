import fastf1
import pandas as pd
import numpy as np
import joblib

fastf1.Cache.enable_cache("f1_cache")

model = joblib.load("f1_model.pkl")

def get_pace(session, driver):
    laps = session.laps.pick_driver(driver)
    laps = laps.pick_quicklaps()

    if laps.empty:
        return np.nan

    return laps["LapTime"].dt.total_seconds().median()


def get_driver_features(fp1, fp2, quali, driver, weather, year):

    quali_results = quali.results.set_index("Abbreviation")

    if driver not in quali_results.index:
        return None

    q = quali_results.loc[driver]

    fp1_pace = get_pace(fp1, driver)
    fp2_pace = get_pace(fp2, driver)

    air = weather["AirTemp"].mean() if "AirTemp" in weather else np.nan
    track = weather["TrackTemp"].mean() if "TrackTemp" in weather else np.nan
    rain = weather["Rainfall"].max() if "Rainfall" in weather else 0

    return {
        "year": year,

        "fp1_pace": fp1_pace,
        "fp2_pace": fp2_pace,
        "pace_dif": fp1_pace - fp2_pace if pd.notnull(fp1_pace) and pd.notnull(fp2_pace) else 0,

        "quali_pos": q["Position"],
        "team": q["TeamName"],

        "air_temp": air,
        "track_temp": track,
        "rainfall": rain
    }

YEAR = 2026
TRACK = "Monaco"

fp1 = fastf1.get_session(YEAR, TRACK, "FP1")
fp2 = fastf1.get_session(YEAR, TRACK, "FP2")
quali = fastf1.get_session(YEAR, TRACK, "Q")

fp1.load(weather=True)
fp2.load(weather=True)
quali.load(weather=True)

weather = quali.weather_data

drivers = fp1.results["Abbreviation"].unique()

rows = []

for d in drivers:
    feats = get_driver_features(fp1, fp2, quali, d, weather, YEAR)
    if feats:
        rows.append(feats)

live_df = pd.DataFrame(rows)

live_df = pd.get_dummies(live_df, columns=["team"], dummy_na=True)

model_features = model.feature_names_in_

live_df = live_df.reindex(columns=model_features, fill_value=0)

# safety cleanup
live_df = live_df.replace([np.inf, -np.inf], np.nan).fillna(0)

live_df["predicted_finish"] = model.predict(live_df)

live_df = live_df.sort_values("predicted_finish")

live_df["win_score"] = 1 / (live_df["predicted_finish"] + 1e-6)

print("\n🏁 LIVE MONACO PREDICTION 🏁\n")

print(live_df[["predicted_finish", "win_score"]].head(10))
