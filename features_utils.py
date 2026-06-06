import pandas as pd
import numpy as np

constructor_points = {
    "Red Bull Racing": 57,
    "Ferrari": 147,
    "McLaren": 106,
    "Mercedes": 219,
    "Aston Martin": 0,
    "Williams": 2,
    "Cadillac": 0,
    "Haas": 19,
    "Alpine": 31,
    "Racing Bulls": 31,
    "Audi": 2
}

driver_points = {
        "ANT": 131,
        "RUS": 88,
        "LEC": 75,
        "HAM": 72,
        "NOR": 58,
        "PIA": 48,
        "VER": 43,
        "GAS": 20,
        "BEA": 18,
        "LAW": 16,
        "COL": 15,
        "HAD": 14,
        "SAI": 6,
        "LIN": 5,
        "BOR": 2,
        "OCO": 1,
        "ALB": 1,
        "ALO": 0,
        "STR": 0,
        "BOT": 0,
        "PER": 0,
        "HUL": 0
    }


def get_pace(session, driver):

    laps = session.laps.pick_driver(driver)
    laps = laps.pick_quicklaps()

    if laps.empty:
        return np.nan

    return laps["LapTime"].dt.total_seconds().median()


def get_driver_features(fp1, fp2, weather, driver, year, fp3=None, quali=None, race=None, training=False):
    """
    Gets all the necessary driver features
    """

    results = fp1.results.set_index("Abbreviation")

    if driver not in results.index:
        return None

    d = results.loc[driver]

    fp1_pace = get_pace(fp1, driver)
    fp2_pace = get_pace(fp2, driver)

    fp3_pace = np.nan
    if fp3:
        fp3_pace = get_pace(fp3, driver)

    quali_results = quali.results.set_index("Abbreviation")

    row = {
        "year": year,
        "driver": driver,

        "fp1_pace": fp1_pace,
        "fp2_pace": fp2_pace,
        "fp3_pace": fp3_pace,
        "pace_dif_1": fp1_pace - fp2_pace if pd.notnull(fp1_pace) and pd.notnull(fp2_pace) else np.nan,
        "pace_dif_2": fp2_pace - fp3_pace if pd.notnull(fp2_pace) and pd.notnull(fp3_pace) else np.nan,

        "driver_points": driver_points[driver],
        "constructor_points": constructor_points.get(d["TeamName"]),

        "quali_pos": quali_results.loc[driver]["Position"],

        "team": d["TeamName"],

        "air_temp": weather["AirTemp"].mean(),
        "track_temp": weather["TrackTemp"].mean(),
        "rainfall": weather["Rainfall"].max(),
    }

    # if training, use more available information

    if training:
        race_results = race.results.set_index("Abbreviation")

        row["grid_pos"] = race_results.loc[driver]["GridPosition"]

        row["finish_pos"] = race_results.loc[driver]["Position"]

    return row
