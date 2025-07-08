import numpy as np
from scipy.signal import welch, csd
import os
import func_helper as fh


def processM1file(SETTINGS, filename, DELAY=1.221, SENS=None, GAIN=None):
    data = np.loadtxt(filename, delimiter=",", skiprows=1)
    if SENS is not None and GAIN is not None:
        print("Calibrating data...")
        data[:, 0] = GAIN[0] * data[:, 0] / SENS[0] 
        data[:, 1] = GAIN[1] * data[:, 1] / SENS[1]
    # Apply delay correction to response channel
    if DELAY is not None and DELAY != 0:
        SAMPLE_RATE = SETTINGS[0]
        delay_samples = int(round(DELAY * 1e-3 * SAMPLE_RATE))
        print(delay_samples)
        if delay_samples > 0:
            data[:, 1] = np.roll(data[:, 1], -delay_samples)
            data[-delay_samples:, 1] = 0  # Zero-pad the end
        elif delay_samples < 0:
            data[:, 1] = np.roll(data[:, 1], -delay_samples)
            data[:-delay_samples, 1] = 0  # Zero-pad the start
    return processNoiseData(SETTINGS, data)

def processM2file(SETTINGS, filename, THRESHOLD, DELAY=1.221, SENS=None, GAIN=None):
    data = np.loadtxt(filename, delimiter=",", skiprows=1)
    if SENS is not None and GAIN is not None:
        print("Calibrating data...")
        data[:, 0] = GAIN[0] * data[:, 0] / SENS[0] 
        data[:, 1] = GAIN[1] * data[:, 1] / SENS[1]
        THRESHOLD = GAIN[1] * THRESHOLD / SENS[1]
    # Apply delay correction to response channel
    if DELAY is not None and DELAY != 0:
        SAMPLE_RATE = SETTINGS[0]
        delay_samples = int(round(DELAY * 1e-3 * SAMPLE_RATE))
        if delay_samples > 0:
            data[:, 1] = np.roll(data[:, 1], -delay_samples)
            data[-delay_samples:, 1] = 0  # Zero-pad the end
        elif delay_samples < 0:
            data[:, 1] = np.roll(data[:, 1], -delay_samples)
            data[:-delay_samples, 1] = 0  # Zero-pad the start
    return processImpactData(SETTINGS, data, THRESHOLD)

def processM3file(SETTINGS, filename, THRESHOLD, SENS=None, GAIN=None):
    data = np.loadtxt(filename, delimiter=",", skiprows=1)
    if SENS is not None and GAIN is not None:
        print("Calibrating data...")
        data = GAIN * data / SENS
        THRESHOLD = GAIN * THRESHOLD / SENS

    return processPendulumData(SETTINGS, data, THRESHOLD)

def processNoiseData(SETTINGS, data):
    SAMPLE_RATE, RESOLUTION, OVERLAP, _, _ = SETTINGS

    WINDOW_SAMPLES = SAMPLE_RATE // RESOLUTION
    OVERLAP_SAMPLES = int(WINDOW_SAMPLES * OVERLAP)
    
    x = data[:, 0]  # Channel 1 (force)
    y = data[:, 1]  # Channel 2 (response)
    
    t = np.arange(len(x)) / SAMPLE_RATE


    f, Pxx = welch(x, fs=SAMPLE_RATE, window='hann', nperseg=WINDOW_SAMPLES,
                   noverlap=OVERLAP_SAMPLES, nfft=WINDOW_SAMPLES, scaling='density')
    f, Pyy = welch(y, fs=SAMPLE_RATE, window='hann', nperseg=WINDOW_SAMPLES,
                   noverlap=OVERLAP_SAMPLES, nfft=WINDOW_SAMPLES, scaling='density')
    f, Pxy = csd(x, y, fs=SAMPLE_RATE, window='hann', nperseg=WINDOW_SAMPLES,
                 noverlap=OVERLAP_SAMPLES, nfft=WINDOW_SAMPLES, scaling='density')
    
    return f, Pxx, Pyy, Pxy, t, x, y

def processImpactData(SETTINGS, data, THRESHOLD):
    SAMPLE_RATE, RESOLUTION, _, BEFORE, AFTER = SETTINGS

    WINDOW_BEFORE = int(BEFORE * SAMPLE_RATE)
    WINDOW_AFTER = int(AFTER * SAMPLE_RATE)
    IMPACT_WINDOW_SIZE = WINDOW_BEFORE + WINDOW_AFTER
    MIN_IMPACT_SEPARATION = int(SAMPLE_RATE/2)
    FFT_SIZE = int(SAMPLE_RATE // RESOLUTION)

    x = data[:, 0]
    y = data[:, 1]
    t = np.arange(len(x)) / SAMPLE_RATE

    # Detect impacts in the response channel
    impact_indices = []
    i = WINDOW_BEFORE
    while i < len(y) - WINDOW_AFTER:
        if np.abs(y[i]) > THRESHOLD:
            impact_indices.append(i)
            i += MIN_IMPACT_SEPARATION
        else:
            i += 1

    total_impacts = len(impact_indices)
    included_indices = impact_indices  # Use all

    t_win = np.linspace(-BEFORE, AFTER, IMPACT_WINDOW_SIZE)
    x_win = []
    y_win = []
    for idx in included_indices:
        force_window = x[idx - WINDOW_BEFORE:idx + WINDOW_AFTER]
        response_window = y[idx - WINDOW_BEFORE:idx + WINDOW_AFTER]
        if len(force_window) == IMPACT_WINDOW_SIZE and len(response_window) == IMPACT_WINDOW_SIZE:
            x_win.append(force_window)
            y_win.append(response_window)

    f = np.fft.rfftfreq(FFT_SIZE, d=1 / SAMPLE_RATE) 
    X_list = []
    Y_list = [] 
    H_list = [] 
    for force_window, response_window in zip(x_win, y_win):
        force_padded = np.pad(force_window, (0, FFT_SIZE - IMPACT_WINDOW_SIZE), mode="constant")
        response_padded = np.pad(response_window, (0, FFT_SIZE - IMPACT_WINDOW_SIZE), mode="constant")

        X = np.fft.rfft(force_padded) / FFT_SIZE
        Y = np.fft.rfft(response_padded) / FFT_SIZE
        H = Y / X

        X_list.append(X)
        Y_list.append(Y)
        H_list.append(H)

    print(f"Detected {total_impacts} impacts, analysed {len(included_indices)} impacts")
    return f, X_list, Y_list, H_list, t, x, y, t_win, x_win, y_win, impact_indices, total_impacts

def processPendulumData(SETTINGS, y, THRESHOLD):
    SAMPLE_RATE, RESOLUTION, _, BEFORE, AFTER = SETTINGS

    WINDOW_BEFORE = int(BEFORE * SAMPLE_RATE)
    WINDOW_AFTER = int(AFTER * SAMPLE_RATE)
    IMPACT_WINDOW_SIZE = WINDOW_BEFORE + WINDOW_AFTER
    MIN_IMPACT_SEPARATION = int(SAMPLE_RATE / 2)
    FFT_SIZE = int(SAMPLE_RATE // RESOLUTION)

    t = np.arange(len(y)) / SAMPLE_RATE

    # Detect impacts
    impact_indices = []
    i = WINDOW_BEFORE
    while i < len(y) - WINDOW_AFTER:
        if np.abs(y[i]) > THRESHOLD:
            impact_indices.append(i)
            i += MIN_IMPACT_SEPARATION
        else:
            i += 1

    total_impacts = len(impact_indices)
    included_indices = impact_indices  # Use all

    t_win = np.linspace(-BEFORE, AFTER, IMPACT_WINDOW_SIZE)
    y_win = []
    for idx in included_indices:
        response_window = y[idx - WINDOW_BEFORE:idx + WINDOW_AFTER]
        if len(response_window) == IMPACT_WINDOW_SIZE:
            y_win.append(response_window)

    f = np.fft.rfftfreq(FFT_SIZE, d=1 / SAMPLE_RATE)
    Y_list = []
    for response_window in y_win:
        response_padded = np.pad(response_window, (0, FFT_SIZE - IMPACT_WINDOW_SIZE), mode="constant")
        Y = np.fft.rfft(response_padded) / FFT_SIZE
        Y_list.append(np.abs(Y))

    print(f"Detected {total_impacts} impacts, analysed {len(included_indices)} impacts")
    return f, Y_list, t, y, t_win, y_win, impact_indices, total_impacts

#####################################################################


def processM1folder(SETTINGS, folder_path, DELAY=1.1221, SENS=None, GAIN=None):
    # Get the list of filenames and sort them
    filenames = sorted(os.listdir(folder_path))
    
    # Initialize lists to store results
    freq = []
    h1_list = []
    h2_list = []
    coh_list = []
    
    for filename in filenames:
        if filename.endswith(".csv"):
            file_path = os.path.join(folder_path, filename)
            f, Pxx, Pyy, Pxy, t, x, y = processM1file(SETTINGS, file_path, DELAY, SENS, GAIN)
            h1, h2, coh = fh.computeTransferCoherence(Pxx, Pyy, Pxy)

            if len(freq) == 0:
                freq = f

            h1_list.append(h1)
            h2_list.append(h2)
            coh_list.append(coh)
    
    return freq, h1_list, h2_list, coh_list


def legacy_loadData(folder_path):
    COH2_frequency = None
    COH3_coherence = []
    FRF3_realfrf = []
    FRF4_imagfrf = []

    # Get the list of filenames and sort them
    filenames = sorted(os.listdir(folder_path))
    
    for filename in filenames:
        if filename.startswith("COH") and filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r') as file:
                lines = file.readlines()
                row_second_column = []
                row_third_column = []
                # Extract the numbers from the second and third columns between lines 84 and 6484
                for line in lines[83:3284]:  # line 84 is index 83
                    columns = line.split()
                    if len(columns) > 2:
                        number_second_column = float(columns[1])
                        number_third_column = float(columns[2])
                        row_second_column.append(number_second_column)
                        row_third_column.append(number_third_column)
                if COH2_frequency is None:
                    COH2_frequency = row_second_column
                COH3_coherence.append(row_third_column)
        
        elif filename.startswith("FRF") and filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r') as file:
                lines = file.readlines()
                row_third_column = []
                row_fourth_column = []
                # Extract the numbers from the third and fourth columns between lines 84 and 6484
                for line in lines[83:6484]:  # line 84 is index 83
                    columns = line.split()
                    if len(columns) > 3:
                        number_third_column = float(columns[2])
                        number_fourth_column = float(columns[3])
                        row_third_column.append(number_third_column)
                        row_fourth_column.append(number_fourth_column)
                FRF3_realfrf.append(row_third_column)
                FRF4_imagfrf.append(row_fourth_column)
    
    # Convert the lists of lists to numpy arrays for better handling
    COH2_frequency = np.array(COH2_frequency)
    COH3_coherence = np.array(COH3_coherence)
    FRF3_realfrf = np.array(FRF3_realfrf)
    FRF4_imagfrf = np.array(FRF4_imagfrf)
    
    return (COH2_frequency, COH3_coherence,
            FRF3_realfrf, FRF4_imagfrf)