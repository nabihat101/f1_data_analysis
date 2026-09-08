import fastf1
import pandas as pd
import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from features_utils import get_driver_features

rows = []

user_track = input("Enter the track you want to train the model on: ")

for year in [2018, 2019, 2021, 2022, 2023, 2024, 2025]:

    # get all the sessions
    race = fastf1.get_session(year, user_track, "R")
    quali = fastf1.get_session(year, user_track, "Q")
    fp1 = fastf1.get_session(year, user_track, "FP1")
    fp2 = fastf1.get_session(year, user_track, "FP2")

    # load all sessions
    fp1.load()
    fp2.load()
    quali.load()
    race.load(weather=True)

    # in case weather is not loaded/race cancelled
    try:
        weather = quali.weather_data
    except:
        weather = pd.DataFrame({
            "AirTemp": [np.nan],
            "TrackTemp": [np.nan],
            "Rainfall": [0]
        })

    for d in race.results["Abbreviation"]:

        feats = get_driver_features(fp1, fp2, weather, d, year, quali, race, training=True)

        if feats is not None:
            rows.append(feats)

df = pd.DataFrame(rows)
df = df.fillna(0)

df = pd.get_dummies(df, columns=["driver", "team"])

train_df = df[df["year"] <= 2023]
test_df = df[df["year"] > 2023]

X_train = train_df.drop("finish_pos", axis=1)
y_train = train_df["finish_pos"]

X_test = test_df.drop("finish_pos", axis=1)
y_test = test_df["finish_pos"]

model = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=3, random_state=42)

model.fit(X_train, y_train)

pred = model.predict(X_test)

joblib.dump(model, f"{user_track}_model.pkl")

print("MAE:", mean_absolute_error(y_test, pred))
