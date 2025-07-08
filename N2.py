import func_helper as fh
import func_analysis as fa
import func_plotting as fp
import numpy as np
import matplotlib.pyplot as plt

filename = "./DATA/A5data/V1a/0_Free/M2_A5_V1_a00.csv"
SETTINGS = (3200, 0.5, 0.90, 0.05, 0.45)
THRESHOLD = 0.1
INCLUDED = None
AtoV = False

f, X_list, Y_list, H_list, t, x, y, t_win, x_win, y_win, impact_indices, total_impacts = fa.processImpactfile(
    SETTINGS, filename, THRESHOLD, INCLUDED
)

fp.plotImpact(SETTINGS, t, x, y, t_win, x_win, y_win, f, X_list, Y_list, H_list, impact_indices, INCLUDED)