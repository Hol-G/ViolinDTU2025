import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from func_helper import *
import numpy as np

def plotNoise(f, t, x, y, Pxx, Pyy, h1, h2, coh, ylim):

    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(2, 3, wspace=0.3, hspace=0.4)

    # Force signal
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(t, x)
    ax1.set_title("Force Signal")
    ax1.set_xlabel("Time [s]")
    ax1.set_ylabel("Force [N]")
    ax1.set_xlim(0, t[-1])
    ax1.grid(True)

    # Response signal
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.plot(t, y)
    ax4.set_title("Response Signal")
    ax4.set_xlabel("Time [s]")
    ax4.set_ylabel("Response [m/s²]")
    ax4.set_xlim(0, t[-1])
    ax4.grid(True)

    # Autospectrum of force signal
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(f, pdB(Pxx))
    ax2.set_title("Autospectrum of Force Signal")
    ax2.set_xlabel("Frequency [Hz]")
    ax2.set_ylabel("Autospectrum [dB]")
    ax2.set_xscale("log")
    ax2.set_xlim(200, f[-1])
    ax2.grid(True)

    # Autospectrum of response signal
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.plot(f, pdB(Pyy))
    ax5.set_title("Autospectrum of Response Signal")
    ax5.set_xlabel("Frequency [Hz]")
    ax5.set_ylabel("Autospectrum [dB]")
    ax5.set_xscale("log")
    ax5.set_xlim(200, f[-1])
    ax5.grid(True)

    # Transfer functions
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(f, dB(h1), label="H1")
    ax3.plot(f, dB(h2), label="H2")
    ax3.set_xscale("log")
    ax3.set_ylim(ylim[0], ylim[1])
    ax3.set_xlim(200, f[-1])
    ax3.set_xlabel("Frequency [Hz]")
    ax3.set_ylabel("TF")
    ax3.legend()
    ax3.set_title("Transfer Functions")
    ax3.grid(True)

    # Coherence
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.plot(f, coh)
    ax6.set_xscale("log")
    ax6.set_xlim(200, f[-1])
    ax6.set_ylim(0, 1)
    ax6.set_xlabel("Frequency [Hz]")
    ax6.set_ylabel("Coherence")
    ax6.set_title("Coherence")
    ax6.grid(True)

    # Adjust layout for consistent alignment
    plt.tight_layout()
    plt.show()

#########################################################

def plotImpact(SETTINGS, t, x, y, t_win, x_win, y_win, f, X_list, Y_list, H_list, impact_indices):


    SAMPLE_RATE, RESOLUTION, OVERLAP, BEFORE, AFTER = SETTINGS

    fig, axs = plt.subplots(2, 4, figsize=(30, 10))
    fig.suptitle("Impact Analysis Results", fontsize=16)

    # --- Column 1: Full time signals with red boxes ---
    axs[0, 0].plot(t, x, label="Force Signal")
    y_min, y_max = axs[0, 0].get_ylim()
    for idx in impact_indices:
        t_start = (idx - int(BEFORE * SAMPLE_RATE)) / SAMPLE_RATE
        t_end = (idx + int(AFTER * SAMPLE_RATE)) / SAMPLE_RATE
        rect = Rectangle((t_start, y_min), t_end - t_start, y_max - y_min,
                         linewidth=1, edgecolor='r', facecolor='none', alpha=0.5)
        axs[0, 0].add_patch(rect)
    axs[0, 0].set_xlabel("Time [s]")
    axs[0, 0].set_ylabel("Force [N]")
    axs[0, 0].set_title("Force Signal with Included Impact Windows")
    axs[0, 0].grid()
    axs[0, 0].legend()

    axs[1, 0].plot(t, y, label="Response Signal")
    y_min, y_max = axs[1, 0].get_ylim()
    for idx in impact_indices:
        t_start = (idx - int(BEFORE * SAMPLE_RATE)) / SAMPLE_RATE
        t_end = (idx + int(AFTER * SAMPLE_RATE)) / SAMPLE_RATE
        rect = Rectangle((t_start, y_min), t_end - t_start, y_max - y_min,
                         linewidth=1, edgecolor='r', facecolor='none', alpha=0.5)
        axs[1, 0].add_patch(rect)
    axs[1, 0].set_xlabel("Time [s]")
    axs[1, 0].set_ylabel("Response [m/s²]")
    axs[1, 0].set_title("Response Signal with Included Impact Windows")
    axs[1, 0].grid()
    axs[1, 0].legend()

    # --- Column 2: Impact windows ---
    for i, (xx, yy) in enumerate(zip(x_win, y_win)):
        axs[0, 1].plot(t_win, xx, label=f"Impact {i}", alpha=0.7)
        axs[1, 1].plot(t_win, yy, label=f"Impact {i}", alpha=0.7)
    axs[0, 1].set_xlabel("Time [s]")
    axs[0, 1].set_ylabel("Force [N]")
    axs[0, 1].set_title("Force Impact Windows")
    axs[0, 1].grid()
    axs[0, 1].legend()
    axs[1, 1].set_xlabel("Time [s]")
    axs[1, 1].set_ylabel("Response [m/s²]")
    axs[1, 1].set_title("Response Impact Windows")
    axs[1, 1].grid()
    axs[1, 1].legend()

    # --- Column 3: Overlay of spectra and transfer functions for each impact ---
    for i, (X_i, Y_i, H_i) in enumerate(zip(X_list, Y_list, H_list)):
        axs[0, 2].plot(f, dB(X_i), alpha=0.5, color='C0')
        axs[0, 2].plot(f, dB(Y_i), alpha=0.5, color='C1')
        axs[1, 2].plot(f, dB(H_i), alpha=0.5, color='C2')
    axs[0, 2].set_xscale("log")
    axs[0, 2].set_xlim(200, SAMPLE_RATE / 2)
    axs[0, 2].set_ylim(-130, -70)
    axs[0, 2].set_xlabel("Frequency [Hz]")
    axs[0, 2].set_ylabel("Magnitude [dB]")
    axs[0, 2].set_title("Impact Spectra (Overlay)")
    axs[0, 2].legend(["Force", "Response"])
    axs[1, 2].set_xscale("log")
    axs[1, 2].set_xlim(200, SAMPLE_RATE / 2)
    axs[1, 2].set_ylim(-30, 30)
    axs[1, 2].set_xlabel("Frequency [Hz]")
    axs[1, 2].set_ylabel("Magnitude [dB]")
    axs[1, 2].set_title("Transfer Functions (Overlay)")

    # --- Column 4: Averaged spectra and transfer function ---
    mean_X = np.mean(np.abs(X_list), axis=0)
    mean_Y = np.mean(np.abs(Y_list), axis=0)
    
    mean_H = np.mean(np.abs(H_list), axis=0)

    axs[0, 3].plot(f, dB(mean_X), label="Force")
    axs[0, 3].plot(f, dB(mean_Y), label="Response")
    axs[0, 3].set_xscale("log")
    axs[0, 3].set_xlim(200, SAMPLE_RATE / 2)
    axs[0, 3].set_ylim(-130, -70)
    axs[0, 3].set_xlabel("Frequency [Hz]")
    axs[0, 3].set_ylabel("Magnitude [dB]")
    axs[0, 3].set_title("Averaged Amplitude Spectra")
    axs[0, 3].legend()

    # --- Averaged Transfer Function Magnitude and Phase (axs[1, 3]) ---
    mag_ax = axs[1, 3]

    # Magnitude (left y-axis)
    mag_ax.plot(f, dB(mean_H), label="Mean H (Magnitude)")
    mag_ax.set_xscale("log")
    mag_ax.set_xlim(200, SAMPLE_RATE / 2)
    mag_ax.set_ylim(-30, 30)
    mag_ax.set_xlabel("Frequency [Hz]")
    mag_ax.set_ylabel("Magnitude [dB]")
    mag_ax.set_title("Averaged Transfer Function")
    mag_ax.legend(loc="upper left")


    # ...existing code...
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


#########################################################

def plotPendulumAnalysis(SETTINGS, f, Y_list, t, y, t_win, y_win, impact_indices):
    """
    Plot pendulum analysis results in a 2x2 grid:
    - Top left: Full time signal with red boxes for included impacts.
    - Bottom left: Overlay of impact windows.
    - Top right: Overlay of spectra for each impact.
    - Bottom right: Averaged spectrum.
    """
    SAMPLE_RATE, RESOLUTION, OVERLAP, BEFORE, AFTER = SETTINGS

    fig, axs = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("Pendulum Impact Analysis", fontsize=16)

    # --- Top left: Full time signal with red boxes ---
    axs[0, 0].plot(t, y, label="Signal")
    y_min, y_max = axs[0, 0].get_ylim()
    for idx in impact_indices:
        t_start = (idx - int(BEFORE * SAMPLE_RATE)) / SAMPLE_RATE
        t_end = (idx + int(AFTER * SAMPLE_RATE)) / SAMPLE_RATE
        rect = Rectangle((t_start, y_min), t_end - t_start, y_max - y_min,
                         linewidth=1, edgecolor='r', facecolor='none', alpha=0.5)
        axs[0, 0].add_patch(rect)
    axs[0, 0].set_xlabel("Time [s]")
    axs[0, 0].set_ylabel("Amplitude")
    axs[0, 0].set_title("Total Signal with Impact Windows")
    axs[0, 0].grid()
    axs[0, 0].legend()

    # --- Bottom left: Overlay of impact windows ---
    for i, yy in enumerate(y_win):
        axs[1, 0].plot(t_win, yy, label=f"Impact {i}", alpha=0.7)
    axs[1, 0].set_xlabel("Time [s]")
    axs[1, 0].set_ylabel("Amplitude")
    axs[1, 0].set_title("Overlayed Impact Windows")
    axs[1, 0].grid()
    axs[1, 0].legend()

    # --- Top right: Overlay of spectra ---
    for Y_i in Y_list:
        axs[0, 1].plot(f, dB(Y_i), alpha=0.5)
    axs[0, 1].set_xscale("log")
    axs[0, 1].set_xlim(200, SAMPLE_RATE / 2)
    axs[0, 1].set_xlabel("Frequency [Hz]")
    axs[0, 1].set_ylabel("Magnitude [dB]")
    axs[0, 1].set_title("Overlayed Spectra")
    axs[0, 1].grid()

    # --- Bottom right: Averaged spectrum ---
    mean_Y = np.mean(Y_list, axis=0)
    axs[1, 1].plot(f, dB(mean_Y), label="Mean Spectrum")
    axs[1, 1].set_xscale("log")
    axs[1, 1].set_xlim(200, SAMPLE_RATE / 2)
    axs[1, 1].set_xlabel("Frequency [Hz]")
    axs[1, 1].set_ylabel("Magnitude [dB]")
    axs[1, 1].set_title("Averaged Spectrum")
    axs[1, 1].grid()
    axs[1, 1].legend()

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

