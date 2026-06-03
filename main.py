# importing libraries

import fastf1
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

fastf1.Cache.enable_cache('cache')

session_2025 = fastf1.get_session(2025, 'Monaco', 'R')

session_2025.load()


