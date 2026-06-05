# importing libraries

import fastf1
import joblib
import pandas as pd
import os
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

os.makedirs("f1_cache", exist_ok=True)
fastf1.Cache.enable_cache("f1_cache")


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
    """
    Returns driver features
    """

    # instead of having indices of (0, 1, ...), set the indices to be driver names
    race_results = race.results.set_index('Abbreviation')
    quali_results = quali.results.set_index('Abbreviation')

    # edge case
    if driver not in race_results.index or driver not in quali_results.index:
        return None

    # get the specific information of each driver
    r_driver = race_results.loc[driver]
    q_driver = quali_results.loc[driver]

    fp1_pace = get_pace(fp1, driver)
    fp2_pace = get_pace(fp2, driver)

    # return the data
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
        "finish_pos": r_driver['Position']
        }

rows = []
for year in [2018, 2019, 2021, 2022, 2023, 2024, 2025]:

    race = fastf1.get_session(year, 'Monaco', 'R')
    quali = fastf1.get_session(year, 'Monaco', 'Q')
    fp1 = fastf1.get_session(year, 'Monaco', 'FP1')
    fp2 = fastf1.get_session(year, 'Monaco', 'FP2')

    fp1.load()
    fp2.load()
    race.load()
    quali.load()

    weather = race.weather_data

    drivers = race.results['Abbreviation'].unique()

    for driver in drivers:
        feats = get_driver_features(fp1, fp2, quali, race, driver, weather, year)

        if feats is not None:
            rows.append(feats)

# cleaning data

df = pd.DataFrame(rows)

df = df.dropna()

df = pd.get_dummies(df, columns=['driver', 'team'])

# split the train and test based on years
train_df = df[df["year"] <= 2023]
test_df = df[df["year"] > 2023]

X_train = train_df.drop(["finish_pos"], axis=1)
y_train = train_df["finish_pos"]

X_test = test_df.drop(["finish_pos"], axis=1)
y_test = test_df["finish_pos"]

# using this model so that it learns and improves score
clf = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=3, random_state=42)

# fitting the model
clf.fit(X_train, y_train)
joblib.dump(clf, "f1_model.pkl")
pred = clf.predict(X_test)

print("MAE:", mean_absolute_error(y_test, pred))


