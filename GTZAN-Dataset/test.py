import sys
from scipy.io import wavfile
from scipy.signal import resample
import numpy as np

np.set_printoptions(threshold=sys.maxsize)

secs = 30
new_sr = 8000

def read_file(file_name):
    samplerate, data = wavfile.read(file_name)
    data = data[0:samplerate * secs]
    data = resample(data, new_sr * secs)

    return data

data = read_file("D:\\Data\\GTZAN-Dataset\\Data\\genres_original\\blues\\blues.00000.wav")
print(data)

