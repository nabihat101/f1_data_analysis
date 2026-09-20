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
    Creates current-race features for one driver.
    """

    fp1_results = fp1.results.set_index("Abbreviation")
    fp2_results = fp2.results.set_index("Abbreviation")
    quali_results = quali.results.set_index("Abbreviation")

    # Make sure driver has qualifying data
    if driver not in quali_results.index:
        return None

    # Get driver's FP1 laps
    fp1_laps = fp1.laps[
        fp1.laps["Driver"] == driver
    ]

    # Get driver's FP2 laps
    fp2_laps = fp2.laps[
        fp2.laps["Driver"] == driver
    ]

    # Make sure practice data exists
    if fp1_laps.empty or fp2_laps.empty:
        return None

    # Get fastest valid practice laps
    fp1_fastest = fp1_laps["LapTime"].dropna().min()
    fp2_fastest = fp2_laps["LapTime"].dropna().min()

    if fp1_fastest is None or fp2_fastest is None:
        return None

    # Get fastest lap in the entire session
    fp1_session_best = fp1.laps["LapTime"].dropna().min()
    fp2_session_best = fp2.laps["LapTime"].dropna().min()

    if fp1_session_best is None or fp2_session_best is None:
        return None

    # Convert to seconds
    fp1_seconds = fp1_fastest.total_seconds()
    fp2_seconds = fp2_fastest.total_seconds()

    fp1_best_seconds = fp1_session_best.total_seconds()
    fp2_best_seconds = fp2_session_best.total_seconds()

    # Gap to fastest driver
    fp1_gap = fp1_seconds - fp1_best_seconds
    fp2_gap = fp2_seconds - fp2_best_seconds

    # How much the driver improved from FP1 to FP2
    fp_pace_change = fp1_seconds - fp2_seconds

    # Qualifying position
    quali_pos = quali_results.loc[driver]["Position"]

    # Get driver's qualifying time
    quali_time = quali_results.loc[driver]["Time"]

    # Calculate gap to pole
    quali_times = quali_results["Time"].dropna()

    if quali_time is not None and len(quali_times) > 0:
        pole_time = quali_times.min()

        quali_gap = (
            quali_time.total_seconds()
            - pole_time.total_seconds()
        )
    else:
        quali_gap = 0

    row = {
        "year": year,
        "track": track,
        "driver": driver,

        "fp1_pace": fp1_seconds,
        "fp2_pace": fp2_seconds,

        "fp1_gap": fp1_gap,
        "fp2_gap": fp2_gap,

        "fp_pace_change": fp_pace_change,

        "quali_pos": quali_pos,
        "quali_gap": quali_gap,

        "team": fp1_results.loc[driver]["TeamName"],

        "air_temp": weather["AirTemp"].mean(),
        "track_temp": weather["TrackTemp"].mean(),
        "rainfall": weather["Rainfall"].max(),
    }

    # Add race result when training
    if training:

        race_results = race.results.set_index("Abbreviation")

        if driver not in race_results.index:
            return None

        row["finish_pos"] = race_results.loc[driver]["Position"]

    return row

def create_empty_history():
    return {
        "driver": {},
        "team": {},
        "driver_track": {},
        "team_track": {}
    }


# to be called on a complete database so that we can map drivers to historical performance and teams to historical performance, as well as driver-track and team-track performance
def build_history(df):
    history = create_empty_history()

    if df.empty:
        return history

    df = df.sort_values(["year", "track"])

    for _, row in df.iterrows():
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

    return history

# returns the the avg of historical performance for drivers and teams at tracks
def get_historical_features(history, driver, team, track):
    driver_history = history["driver"].get(driver, [])
    team_history = history["team"].get(team, [])
    driver_track_history = history["driver_track"].get((driver, track), [])
    team_track_history = history["team_track"].get((team, track), [])

    if len(driver_history) > 0:
        driver_recent_avg = sum(driver_history[-5:]) / len(driver_history[-5:])
    else:
        driver_recent_avg = None

    if len(team_history) > 0:
        team_recent_avg = sum(team_history[-5:]) / len(team_history[-5:])
    else:
        team_recent_avg = None

    if len(driver_track_history) > 0:
        driver_track_avg = sum(driver_track_history) / len(driver_track_history)
    else:
        driver_track_avg = None

    if len(team_track_history) > 0:
        team_track_avg = sum(team_track_history) / len(team_track_history)
    else:
        team_track_avg = None

    return {
        "driver_recent_avg_finish": driver_recent_avg,
        "team_recent_avg_finish": team_recent_avg,
        "driver_track_avg_finish": driver_track_avg,
        "team_track_avg_finish": team_track_avg
    }