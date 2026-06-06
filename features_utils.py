import pandas as pd
import numpy as np

def get_driver_features(fp1, fp2, weather, driver, year, quali=None, race=None, training=False):
    """
    Gets all the necessary driver features
    """

    results = fp1.results.set_index("Abbreviation")

    if driver not in results.index:
        return None

    d = results.loc[driver]
    fp2_results = fp2.results.set_index("Abbreviation")

    fp1_place = results.loc[driver]["Position"]
    fp2_place = fp2_results.loc[driver]["Position"]

    quali_results = quali.results.set_index("Abbreviation")

    row = {
        "year": year,
        "driver": driver,

        "fp1_pace": fp1_place,
        "fp2_pace": fp2_place,

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
