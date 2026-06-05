# importing libraries

import fastf1
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

fastf1.Cache.enable_cache('cache')

rows = []

for year in [2022, 2023, 2024, 2025]:
    race = fastf1.get_session(year, 'Monaco', 'R')
    quali = fastf1.get_session(year, 'Monaco', 'Q')
    race.load()
    quali.load()

    race_results = race.results
    quali_results = quali.results





