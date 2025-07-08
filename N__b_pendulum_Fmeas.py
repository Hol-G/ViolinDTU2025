import func_helper as fh
import func_analysis as fa
import func_plotting as fp
import numpy as np
import matplotlib.pyplot as plt
from func_helper import dB

file_input = "./TESTDATA/M3_ref.csv"

file_Htrue = "./TESTDATA/H_true.csv"
file_Xtrue = "./TESTDATA/X_true.csv"
file_Xest = "./TESTDATA/X_est.csv"

SETTINGS = (3200, 0.5, 0.90, 0.05, 0.45)
THRESHOLD = 0.01

f, Y_list, t, y, t_win, y_win, impact_indices, total_impacts = fa.processPendulumfile(
    SETTINGS, file_input, THRESHOLD
)


Y = fh.accelToVel(f, Y_list)
Y = np.mean(np.abs(Y), axis=0)

Htrue = np.loadtxt(file_Htrue, delimiter=",", skiprows=1, usecols=(1, 2))
Htrue = Htrue[:, 0] + 1j * Htrue[:, 1]

Xtrue = np.loadtxt(file_Xtrue, delimiter=",", skiprows=1, usecols=(1, 2))
Xtrue = Xtrue[:, 0] + 1j * Xtrue[:, 1]


# Smooth the magnitude (in dB) of Xest
Xest = Y / Htrue
Xsmooth = fh.smooth(Xest)
# Save Xest as CSV (real and imaginary parts)
np.savetxt(
    "X_est.csv",
    np.column_stack([f, np.real(Xsmooth), np.imag(Xsmooth)]),
    delimiter=",",
    header="f,Re,Im",
    comments=""
)


Hest = Y / Xsmooth

# PHASES
Xtrue_phase = np.unwrap(np.angle(Xtrue))
Xest_phase_unwrapped = np.unwrap(np.angle(Xsmooth))
Htrue_phase = np.unwrap(np.angle(Htrue))
Hest_phase = np.unwrap(np.angle(Hest))

# ...existing code...

fig, axs = plt.subplots(1, 2, figsize=(12, 5), sharex=True)

# Left: Estimated X (raw), Estimated X (smooth), X_true
axs[0].plot(f, dB(Xest), label="Estimated X (raw)", alpha=0.7)
axs[0].plot(f, dB(Xsmooth), label="Estimated X (smoothed)", linewidth=2)
axs[0].plot(f, dB(Xtrue), label="True X", linestyle='--')
axs[0].set_title("X: Estimated (raw/smooth) and True")
axs[0].set_ylabel("Amplitude [dB]")
axs[0].legend()
axs[0].set_xlim(200, f[-1])
axs[0].set_ylim(-150, -80)
axs[0].grid(True, which="both", linestyle="--", linewidth=0.5)

# Right: Estimated H, True H
axs[1].plot(f, dB(Hest), label="Estimated H")
axs[1].plot(f, dB(Htrue), label="True H", linestyle='--')
axs[1].set_title("H: Estimated and True")
axs[1].set_ylabel("Amplitude / dB")
axs[1].legend()
axs[1].set_xlabel("Frequency / Hz")
axs[1].set_xscale("log")
axs[1].set_xlim(200, f[-1])
axs[1].set_ylim(-100, -40)
axs[1].grid(True, which="both", linestyle="--", linewidth=0.5)

plt.tight_layout()
plt.show()