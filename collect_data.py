import os
import fastf1
import pandas as pd
import numpy as np

from features_utils import get_driver_features, get_historical_features

os.makedirs(".fastf1_cache", exist_ok=True)

YEARS = [2022, 2023, 2024, 2025]
OUTPUT_FILE = "f1_race_data.csv"

fastf1.Cache.enable_cache(".fastf1_cache")

def create_empty_history():
    return {
        "driver": {},
        "team": {},
        "driver_track": {},
        "team_track": {}
    }

# after we've collected a race, we want to update the history with the new data so that we can use it for future races
def update_history(history, race_rows):
    for row in race_rows:
        driver = row["driver"]
        team = row["team"]
        track = row["track"]
        finish = row["finish_pos"]

        if driver not in history["driver"]:
            history["driver"][driver] = []
        history["driver"][driver].append(finish)

        if team not in history["team"]:
            history["team"][team] = []
        history["team"][team].append(finish)

        driver_track_key = (driver, track)
        if driver_track_key not in history["driver_track"]:
            history["driver_track"][driver_track_key] = []
        history["driver_track"][driver_track_key].append(finish)

        team_track_key = (team, track)
        if team_track_key not in history["team_track"]:
            history["team_track"][team_track_key] = []
        history["team_track"][team_track_key].append(finish)


# we have this function to rebuild the history from the existing dataset, so we don't have to recalculate it every time we run the script bc of FastF1s limitations
def rebuild_history(df):
    history = create_empty_history()
    collected_races = set()

    if df.empty:
        return df, history, collected_races

    df = df.drop_duplicates(subset=["year", "track", "driver"], keep="last").copy()

    for year in YEARS:
        print(f"\nRebuilding history for {year}...")

        try:
            schedule = fastf1.get_event_schedule(year)
        except Exception as e:
            print(f"Could not load schedule for {year}: {e}")
            continue

        for _, event in schedule.iterrows():
            if pd.isna(event["RoundNumber"]) or event["RoundNumber"] == 0:
                continue

            track = event["EventName"]
            race_mask = (df["year"] == year) & (df["track"] == track)
            race_indices = df.index[race_mask].tolist()

            if len(race_indices) == 0:
                continue

            print(f"  Rebuilding {year} {track}")

            race_rows = []

            for index in race_indices:
                row = df.loc[index]

                # building historical features for each driver in the race based on the history we've built so far
                historical = get_historical_features(
                    history,
                    row["driver"],
                    row["team"],
                    track
                )

                for key, value in historical.items():
                    df.loc[index, key] = value

                race_rows.append({
                    "driver": row["driver"],
                    "team": row["team"],
                    "track": track,
                    "finish_pos": row["finish_pos"]
                })

            update_history(history, race_rows)
            collected_races.add((year, track))

    return df, history, collected_races

# finds if exisiting dataset exists, if it does, load it and rebuild history, if not, create new dataset
if os.path.exists(OUTPUT_FILE):
    print("\nExisting dataset found.")
    df = pd.read_csv(OUTPUT_FILE)
    print(f"Existing rows: {len(df)}")
else:
    print("\nNo existing dataset found. Starting new dataset.")
    df = pd.DataFrame()


df, history, collected_races = rebuild_history(df)

print(f"\nExisting races: {len(collected_races)}")

rows = df.to_dict("records")


for year in YEARS:
    print(f"\n========== {year} ==========")

    try:
        schedule = fastf1.get_event_schedule(year)
    except Exception as e:
        print(f"Could not load schedule for {year}: {e}")
        continue

    for _, event in schedule.iterrows():
        if pd.isna(event["RoundNumber"]) or event["RoundNumber"] == 0:
            continue

        track = event["EventName"]
        race_key = (year, track)

        if race_key in collected_races:
            print(f"SKIPPING {year} {track} - already collected")
            continue

        print(f"\nCollecting {year} {track}...")

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
            print(f"FAILED {year} {track}: {e}")
            continue

        try:
            weather = quali.weather_data
        except Exception:
            weather = pd.DataFrame({
                "AirTemp": [np.nan],
                "TrackTemp": [np.nan],
                "Rainfall": [0]
            })

        race_rows = []

        for driver in race.results["Abbreviation"]:
            feats = get_driver_features(
                fp1, fp2, weather, driver, year, track, quali, race, True
            )

            if feats is None:
                print(f"  Skipping {driver} - missing data")
                continue

            team = feats["team"]

            historical = get_historical_features(
                history, driver, team, track
            )

            feats.update(historical)
            race_rows.append(feats)

        if len(race_rows) == 0:
            print(f"No usable data for {year} {track}.")
            continue

        rows.extend(race_rows)

        update_history(history, race_rows)
        collected_races.add(race_key)

        df = pd.DataFrame(rows)
        df = df.drop_duplicates(
            subset=["year", "track", "driver"],
            keep="last"
        )
        df.to_csv(OUTPUT_FILE, index=False)

        print(f"COMPLETED {year} {track} ({len(race_rows)} drivers)")
        print(f"Dataset now has {len(df)} rows.")


df = pd.DataFrame(rows)
df = df.drop_duplicates(
    subset=["year", "track", "driver"],
    keep="last"
)
df.to_csv(OUTPUT_FILE, index=False)

print("\n========================================")
print("DATA COLLECTION COMPLETE")
print("========================================")
print(f"Total rows: {len(df)}")
print(f"Total races: {len(df[['year', 'track']].drop_duplicates())}")
print(f"Saved to: {OUTPUT_FILE}")