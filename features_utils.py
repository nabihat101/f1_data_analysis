def get_driver_features(
        fp1,
        fp2,
        weather,
        driver,
        year,
        track,
        quali=None,
        race=None,
        training=False
):
    """
    Creates the features used to predict a driver's race result.
    """

    fp1_results = fp1.results.set_index("Abbreviation")
    fp2_results = fp2.results.set_index("Abbreviation")
    quali_results = quali.results.set_index("Abbreviation")

    if driver not in quali_results.index:
        return None

    # Get FP1 laps for this driver
    fp1_laps = fp1.laps[
        fp1.laps["Driver"] == driver
    ]

    # Get FP2 laps for this driver
    fp2_laps = fp2.laps[
        fp2.laps["Driver"] == driver
    ]

    # Make sure practice data exists
    if fp1_laps.empty or fp2_laps.empty:
        return None

    # Get fastest valid lap
    fp1_fastest = fp1_laps["LapTime"].dropna().min()
    fp2_fastest = fp2_laps["LapTime"].dropna().min()

    if fp1_fastest is None or fp2_fastest is None:
        return None

    driver_info = fp1_results.loc[driver]

    row = {
        "year": year,
        "track": track,
        "driver": driver,

        "fp1_pace": fp1_fastest.total_seconds(),
        "fp2_pace": fp2_fastest.total_seconds(),
        "quali_pos": quali_results.loc[driver]["Position"],

        "team": driver_info["TeamName"],

        "air_temp": weather["AirTemp"].mean(),
        "track_temp": weather["TrackTemp"].mean(),
        "rainfall": weather["Rainfall"].max(),
    }

    # During training, we know the actual race result.
    if training:
        race_results = race.results.set_index("Abbreviation")

        if driver not in race_results.index:
            return None

        row["finish_pos"] = race_results.loc[driver]["Position"]

    return row