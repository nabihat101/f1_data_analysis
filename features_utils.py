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

    # Error handling: Make sure the driver has data from all required sessions
    if driver not in fp1_results.index:
        return None

    if driver not in fp2_results.index:
        return None

    if driver not in quali_results.index:
        return None

    driver_info = fp1_results.loc[driver]

    row = {
        "year": year,
        "track": track,
        "driver": driver,

        "fp1_pos": fp1_results.loc[driver]["Position"],
        "fp2_pos": fp2_results.loc[driver]["Position"],
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