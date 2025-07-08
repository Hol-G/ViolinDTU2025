import func_helper as fh
import func_analysis as fa
import func_plotting as fp
import numpy as np
import json

filename = "./TESTDATA/M1_test.csv"
SETTINGS = (3200, 0.5, 0.90, 0.05, 0.45)

# Import calibration values
with open("calibration_values.json", "r") as f:
    calibration_data = json.load(f)
SENS = calibration_data["SENS"]
GAIN = calibration_data["GAIN"]

# Specify the channels for accelerometer and force transducer
SENS = [SENS[1], SENS[0]]
GAIN = [GAIN[1], GAIN[0]]

f, Pxx, Pyy, Pxy, t, x, y = fa.processM1file(SETTINGS, filename, SENS, GAIN)
h1, h2, coh = fh.computeTransferCoherence(Pxx, Pyy, Pxy)

h1 = fh.accelToVel(f, h1)
h2 = fh.accelToVel(f, h2)

fp.plotNoise(f, t, x, y, Pxx, Pyy, h1, h2, coh, [-50,10])
