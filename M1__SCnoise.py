import func_measure as fm

SETTINGS = (3200, 0.5, 0.90, 0.05, 0.45) # (Sample Rate, Resolution, Overlap, Window Before, Window After)
MAX_AVERAGES = 100
AUDIO_BUFFERSIZE = 256

output_filename = "./M1_noisefile.csv"


fm.noiseSoundcard(SETTINGS, MAX_AVERAGES, AUDIO_BUFFERSIZE, output_filename)