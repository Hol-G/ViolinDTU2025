import func_analysis as fa
import func_plotting as fp

file_input = "./DATA/04b_PendulumValidate/V1b_innate/M3_V1b_04.csv"
SETTINGS = (3200, 0.5, 0.90, 0.05, 0.45)
THRESHOLD = 0.2

f, Y_list, t, y, t_win, y_win, impact_indices, total_impacts = fa.processPendulumfile(
    SETTINGS, file_input, THRESHOLD,
)

fp.plotPendulumAnalysis(SETTINGS, f, Y_list, t, y, t_win, y_win, impact_indices)
