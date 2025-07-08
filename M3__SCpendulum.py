import func_measure as fm

SETTINGS = (3200, 0.5, 0.90, 0.05, 0.45) # (Sample Rate, Resolution, Overlap, Window Before, Window After)
IMPACT_THRESHOLD = 0.05
IMPACT_COUNT = 10
AUDIO_BUFFERSIZE = 256

output_filename = "./M3_pendulumfile.csv"


fm.pendulumSoundcard(SETTINGS, IMPACT_THRESHOLD, IMPACT_COUNT, AUDIO_BUFFERSIZE, output_filename)

