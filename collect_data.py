import fastf1
import pandas as pd
import numpy as np
from features_utils import get_driver_features

YEARS = [2018, 2019, 2021, 2022, 2023, 2024, 2025]
rows = []

for year in YEARS:

    # get races for that year
    schedule = fastf1.get_event_schedule(year)

    for _, event in schedule.iterrows():
        track = event["EventName"]

        # Skip events that aren't actual races
        if pd.isna(event["RoundNumber"]) or event["RoundNumber"] == 0:
            continue

        try:
            race = fastf1.get_session(year, track, "R")
            quali = fastf1.get_session(year, track, "Q")
            fp1 = fastf1.get_session(year, track, "FP1")
            fp2 = fastf1.get_session(year, track, "FP2")

            fp1.load()
            fp2.load()
            quali.load()
            race.load(weather=True)

        except Exception as e:
            print(f"    Skipping {track}: {e}")
            continue

        # Get weather data
        try:
            weather = quali.weather_data

        except Exception:
            weather = pd.DataFrame({
                "AirTemp": [np.nan],
                "TrackTemp": [np.nan],
                "Rainfall": [0]
            })
        
        # Create one row for every driver
        for driver in race.results["Abbreviation"]:

            feats = get_driver_features(
                fp1=fp1,
                fp2=fp2,
                weather=weather,
                driver=driver,
                year=year,
                track=track,
                quali=quali,
                race=race,
                training=True
            )

            if feats is not None:
                rows.append(feats)

# Convert everything into a DataFrame
df = pd.DataFrame(rows)

# Save the dataset
df.to_csv("f1_race_data.csv", index=False)

print("\nFinished collecting data!")
print(f"Total rows: {len(df)}")
print(f"Total columns: {len(df.columns)}")

print("\nColumns:")
print(df.columns.tolist())