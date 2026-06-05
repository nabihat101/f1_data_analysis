# importing libraries

import fastf1
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

os.makedirs("f1_cache", exist_ok=True)
fastf1.Cache.enable_cache("f1_cache")

rows = []

for year in [2018, 2019, 2021, 2022, 2023, 2024, 2025]:

    # get the sessions and load them for each year
    race = fastf1.get_session(year, 'Monaco', 'R')
    quali = fastf1.get_session(year, 'Monaco', 'Q')
    fp1 = fastf1.get_session(year, 'Monaco', 'FP1')
    fp2 = fastf1.get_session(year, 'Monaco', 'FP2')

    fp1.load()
    fp2.load()
    race.load()
    quali.load()

    # instead of having indices of (0, 1, ...), set the indices to be driver names
    race_results = race.results.set_index('Abbreviation')
    quali_results = quali.results.set_index('Abbreviation')

    # loop through all drivers
    for driver in race_results.index:

        if driver not in quali_results.index:
            continue

        # get the specific information of each driver
        r_driver = race_results.loc[driver]
        q_driver = quali_results.loc[driver]

        # create a row for our new table
        row = {
            "year": year,
            "driver": driver,

            "quali_pos": q_driver['Position'],
            "grid_pos": r_driver['GridPosition'],
            "team": r_driver['TeamName'],
            "finish_pos": r_driver['Position']
        }

        rows.append(row)

# cleaning data
df = pd.DataFrame(rows)

df = df.dropna()

df = pd.get_dummies(df, columns=['driver', 'team'])

# Create X (features )
x = df.drop("finish_pos", axis=1)

# Create y (labels)
y = df["finish_pos"]

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
pred = clf.predict(X_test)

print("MAE:", mean_absolute_error(y_test, pred))




