import fastf1
import pandas as pd
from features_utils import get_driver_features

YEARS = [2022, 2023, 2024, 2025]

# Load existing data if it exists
try:
    df_existing = pd.read_csv("f1_race_data.csv")
    rows = df_existing.to_dict("records")

    print(f"Loaded {len(rows)} existing rows.")

except FileNotFoundError:
    rows = []
    print("No existing dataset found. Starting from scratch.")


for year in YEARS:

    print(f"\nCollecting data for {year}...")

    schedule = fastf1.get_event_schedule(year)

    for _, event in schedule.iterrows():

        track = event["EventName"]

        # Skip events that aren't actual races
        if pd.isna(event["RoundNumber"]) or event["RoundNumber"] == 0:
            continue

        # Check whether this race has already been collected
        already_collected = any(
            row["year"] == year and row["track"] == track
            for row in rows
        )

        if already_collected:
            print(f"  Skipping {track} - already collected")
            continue

        print(f"  Collecting {track}...")

        # Load sessions
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
                "AirTemp": [float("nan")],
                "TrackTemp": [float("nan")],
                "Rainfall": [0]
            })

        # Create one row for every driver
        race_rows = []

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
                race_rows.append(feats)

        # Add this race's data to our overall dataset
        rows.extend(race_rows)

        # SAVE IMMEDIATELY after this race
        df = pd.DataFrame(rows)
        df.to_csv("f1_race_data.csv", index=False)

        print(f"    Saved {len(race_rows)} drivers.")
        print(f"    Total rows: {len(rows)}")


print("\nFinished collecting data!")

df = pd.DataFrame(rows)

print(f"Total rows: {len(df)}")
print(f"Total columns: {len(df.columns)}")

print("\nColumns:")
print(df.columns.tolist())