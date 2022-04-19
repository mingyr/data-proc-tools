import os, sys
import librosa
from scipy.signal import resample
from glob import glob
import numpy as np
import tensorflow as tf
from random import shuffle

data_root = "D:\\Data\\GTZAN-Dataset\\Data\\genres_original"

# list file candidates

train_files = []
val_files = []
test_files = []

train_slice = slice(0, 80)
val_slice = slice(80, 90)
test_slice = slice(90, 100)

for label, name in enumerate(os.listdir(data_root)):
    # print(f"{name} - {label}")
    full_data_path = os.path.join(data_root, name, "*.wav")
    files = glob(full_data_path)
    files = [(f, label) for f in files]
    train_files.extend(files[train_slice])
    val_files.extend(files[val_slice])
    test_files.extend(files[test_slice])    

print("{} files for training".format(len(train_files)))
print("{} files for validation".format(len(val_files)))
print("{} files for testing".format(len(test_files)))

shuffle(train_files)
shuffle(val_files)
shuffle(test_files)

secs = 30
new_sr = 8000
num_steps = 6
num_chans = 1
seg_dur = 5

def read_file(file_name):
    data, samplerate = librosa.load(file_name)
    data = data[0:samplerate * secs]
    data = resample(data, new_sr * secs)

    return data

def float_feature(value):
    return tf.train.Feature(float_list = tf.train.FloatList(value = value))

def bytes_feature(value):
    return tf.train.Feature(bytes_list = tf.train.BytesList(value = [value]))

def int64_feature(value):
    return tf.train.Feature(int64_list = tf.train.Int64List(value = [value]))

outdir = "D:\\Data\\GTZAN-Dataset"
assert os.path.exists(outdir), "Invalid output directory: {}".format(outdir)
for i, t in enumerate(["train", "val", "test"]):
    tfrec_filename = os.path.join(outdir, f"music-{t}.tfr")

    writer = tf.python_io.TFRecordWriter(tfrec_filename)

    if i == 0:
        files = train_files
    elif i == 1:
        files = val_files
    else:
        files = test_files
        
    for f in files:
        ff = f[0]
        lb = f[1]

        print("current processing data file: {}".format(ff))
        
        try:        
            data = read_file(ff)
        except:
            print("error for processing data file: {}".format(ff))
            continue
        
        data0 = data[(new_sr * 5 * 0):(new_sr * 5 * 1)].astype(np.float32)
        data1 = data[(new_sr * 5 * 1):(new_sr * 5 * 2)].astype(np.float32)
        data2 = data[(new_sr * 5 * 2):(new_sr * 5 * 3)].astype(np.float32)
        data3 = data[(new_sr * 5 * 3):(new_sr * 5 * 4)].astype(np.float32)
        data4 = data[(new_sr * 5 * 4):(new_sr * 5 * 5)].astype(np.float32)
        data5 = data[(new_sr * 5 * 5):(new_sr * 5 * 6)].astype(np.float32)
        
        data0 = np.reshape(data0, (-1, new_sr)).transpose()
        data1 = np.reshape(data1, (-1, new_sr)).transpose()
        data2 = np.reshape(data2, (-1, new_sr)).transpose()
        data3 = np.reshape(data3, (-1, new_sr)).transpose()
        data4 = np.reshape(data4, (-1, new_sr)).transpose()
        data5 = np.reshape(data5, (-1, new_sr)).transpose()
        
        waves = [data0, data1, data2, data3, data4, data5]
        
        wave_features = [float_feature(np.reshape(w, [-1])) for w in waves]
        label_features = [int64_feature(lb)]
        name_features = [bytes_feature(str.encode(os.path.basename(ff)))]
        
        feature_list = {
            'music': tf.train.FeatureList(feature = wave_features),
            'genre': tf.train.FeatureList(feature = label_features),
            'source': tf.train.FeatureList(feature = name_features)
        }

        feature_lists = tf.train.FeatureLists(feature_list = feature_list)
        example = tf.train.SequenceExample(feature_lists = feature_lists)

        writer.write(example.SerializeToString())

    writer.close()
