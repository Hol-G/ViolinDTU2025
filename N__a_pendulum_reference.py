import func_helper as fh
import func_analysis as fa
import func_plotting as fp
import numpy as np

file_input = "./TESTDATA/M2_ref.csv"
file_Htrue = "./TESTDATA/H_true.csv"
file_Xtrue = "./TESTDATA/X_true.csv"

SETTINGS = (3200, 0.5, 0.90, 0.05, 0.45)
THRESHOLD = 0.01

f, X_list, Y_list, H_list, t, x, y, t_win, x_win, y_win, impact_indices, total_impacts = fa.processImpactfile(
    SETTINGS, file_input, THRESHOLD
)
X_list = np.abs(X_list)
Y_list = np.abs(Y_list)

X, Y, H = np.mean(X_list, axis=0), np.mean(Y_list, axis=0), np.mean(H_list, axis=0)
H = fh.accelToVel(f, H)

fp.plotImpact(SETTINGS, t,x,y, t_win,x_win,y_win, f,X_list,Y_list,H_list, impact_indices)

np.savetxt(
    file_Htrue,
    np.column_stack([f, H.real, H.imag]),
    delimiter=",",
    header="Frequency, H_real, H_imag",
    comments=""
)

np.savetxt(
    file_Xtrue,
    np.column_stack([f, X.real, X.imag]),
    delimiter=",",
    header="Frequency, X_real, X_imag",
    comments=""
)

