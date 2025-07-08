import func_calib as fc
import json

#       ACCEL       FORCE           LASER   
SENS = [3.16e-3,    316e-3,    10000/500]
GAIN = [3.369,       0.48,         2.765]
MASS = 0.1085

accel_channel = 0
#fc.calibrateAccelerometer(accel_channel, SENS, GAIN)

force_channel = 0
#fc.calibrateForcetransducer(force_channel, accel_channel, SENS, GAIN, MASS)

laser_channel = 1
#fc.calibrateLaser(laser_channel, accel_channel, SENS, GAIN)

mic_sens = 51.46e-3
mic_gain = 4.316

mic_level = 94
mic_freq = 1000
mic_channel = 1
#fc.calibrateMicrophone(mic_channel, mic_sens, mic_gain, mic_level, mic_freq)

calibration_data = {
    "SENS": SENS,
    "GAIN": GAIN,
    "MASS": MASS,
    "accel_channel": accel_channel,
    "force_channel": force_channel,
    "laser_channel": laser_channel,
    "MIC_SENS": mic_sens,
    "MIC_GAIN": mic_gain
}

with open("calibration_values.json", "w") as f:
    json.dump(calibration_data, f, indent=4)