# importing libraries

import fastf1
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

fastf1.Cache.enable_cache('cache')

rows = []

for year in [2022, 2023, 2024, 2025]:

    # get the sessions and load them for each year
    race = fastf1.get_session(year, 'Monaco', 'R')
    quali = fastf1.get_session(year, 'Monaco', 'Q')
    race.load()
    quali.load()

    # instead of having indices of (0, 1, ...), set the indices to be driver names
    race_results = race.results.set_index('Abbreviation')
    quali_results = quali.results.set_index('Abbreviation')

    # loop through all drivers
    for driver in race_results.index:

        # get the specific information of each driver
        r_driver = race_results.loc[driver]
        q_driver = quali_results.loc[driver]

        # create a row for our new table
        row = {
            "year": year,
            "driver": driver,

            "quali_pos": q_driver['Position'],
            "grid_pos": r_driver['GridPosition'],
            "team": r_driver['Team'],
            "finish_pos": r_driver['Position']
        }

        rows.append(row)




