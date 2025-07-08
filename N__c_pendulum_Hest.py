import func_helper as fh
import func_analysis as fa
import func_plotting as fp
import numpy as np
import matplotlib.pyplot as plt
from func_helper import dB

def wrap_phase_deg(phase_deg):
    """Wrap phase in degrees to [-180, 180]."""
    return (phase_deg + 180) % 360 - 180



file_input = "./TESTDATA/M3_test.csv"
file_Htrue = "./TESTDATA/H_true.csv"
file_Xest = "./TESTDATA/X_est.csv"

SETTINGS = (3200, 0.5, 0.90, 0.05, 0.45)
THRESHOLD = 0.01
INCLUDED = None 

f, Y_list, t, y, t_win, y_win, impact_indices, total_impacts = fa.processPendulumfile(
    SETTINGS, file_input, THRESHOLD, INCLUDED
)

Y = np.mean(Y_list, axis=0)
Y = fh.accelToVel(f, Y)

Htrue = np.loadtxt(file_Htrue, delimiter=",", skiprows=1, usecols=(1, 2))
Htrue = Htrue[:, 0] + 1j * Htrue[:, 1]
Xest = np.loadtxt(file_Xest, delimiter=",", skiprows=1, usecols=(1, 2))
Xest = Xest[:, 0] + 1j * Xest[:, 1]
Hest = Y / Xest

fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# Magnitude subplot
axs[0].plot(f, dB(Htrue), label="True H", linestyle='--')
axs[0].plot(f, dB(Hest), label="Estimated H")
axs[0].set_xscale("log")
axs[0].set_xlim(200, f[-1])
axs[0].set_ylabel("Magnitude [dB]")
axs[0].set_title("Pendulum System Frequency Response (H)")
axs[0].legend()
axs[0].grid(True, which="both", linestyle="--", linewidth=0.5)

# Phase subplot
Htrue_phase = np.degrees(np.unwrap(np.angle(Htrue)))
Hest_phase = np.degrees(np.unwrap(np.angle(Hest)))
Htrue_phase_wrapped = wrap_phase_deg(Htrue_phase)
Hest_phase_wrapped = wrap_phase_deg(Hest_phase)

axs[1].plot(f, Htrue_phase_wrapped, label="True H phase", linestyle='--')
axs[1].plot(f, Hest_phase_wrapped, label="Estimated H phase")
axs[1].set_xscale("log")
axs[1].set_xlim(200, f[-1])
axs[1].set_xlabel("Frequency (Hz)")
axs[1].set_ylabel("Phase [deg]")
axs[1].set_title("Pendulum System Phase Response (H)")
axs[1].legend()
axs[1].grid(True, which="both", linestyle="--", linewidth=0.5)

plt.tight_layout()
plt.show()