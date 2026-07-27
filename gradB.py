import numpy as np
import pandas as pd
import sys
sys.path.insert(0, "/home/leeviloi/analysator-dev")
import analysator as pt; print(pt.__file__)

"""
Finding magnetic field grandients according to paper:
https://angeo.copernicus.org/articles/43/115/2025/angeo-43-115-2025.pdf
and comparing it to true Vlasiator gradients
"""
def calculate_gradB():
    gradB = np.empty(3)
    return gradB


