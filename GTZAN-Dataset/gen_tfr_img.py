import os, sys
import librosa
import librosa.display
from scipy.signal import resample
from glob import glob
import numpy as np
import tensorflow as tf
from random import shuffle
from matplotlib import pyplot as plt
from io import BytesIO
import cv2

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

def read_file(file_name, fig):
    data, samplerate = librosa.load(file_name)
    data = data[0:samplerate * secs]
    data = resample(data, new_sr * secs)

    images = []
    for i in range(int(secs / seg_dur)):
        stft = librosa.stft(data[new_sr * seg_dur * i : new_sr * seg_dur * (i + 1)])
        stft_db = librosa.amplitude_to_db(abs(stft))
        librosa.display.specshow(stft_db, sr=new_sr) # , x_axis="time", y_axis="hz")

        # update/draw the elements
        # get the width and the height to resize the matrix
        fig.canvas.draw()  
        l, b, w, h = fig.canvas.figure.bbox.bounds
        w, h = int(w), int(h)

        bio = BytesIO()
        fig.savefig(bio, format="rgba")
        bio.seek(0)
        image = np.frombuffer(bio.read(), dtype=np.uint8)
        image = image.reshape(h, w, -1)
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

        # cv2.imshow('image window', image)
        # # add wait key. window waits until user presses a key
        # cv2.waitKey(0)
        # # and finally destroy/close all open windows
        # cv2.destroyAllWindows()

        plt.clf()
        images.append(image)
        
    return images

def float_feature(value):
    return tf.train.Feature(float_list = tf.train.FloatList(value = value))

def bytes_feature(value):
    return tf.train.Feature(bytes_list = tf.train.BytesList(value = [value]))

def int64_feature(value):
    return tf.train.Feature(int64_list = tf.train.Int64List(value = [value]))

outdir = "D:\\Data\\GTZAN-Dataset\\Image"
assert os.path.exists(outdir), "Invalid output directory: {}".format(outdir)

fig = plt.figure(figsize=(3.2, 2.4))
plt.axis("off")
plt.tight_layout(pad=0.0)

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
            data = read_file(ff, fig)
        except:
            print("error for processing data file: {}".format(ff))
            continue        
        
        # data = read_file(ff)
        
        wave_features = [float_feature(np.reshape(w, [-1])) for w in data]
        label_features = [int64_feature(lb)]
        
        feature_list = {
            'music': tf.train.FeatureList(feature = wave_features),
            'genre': tf.train.FeatureList(feature = label_features)
        }

        feature_lists = tf.train.FeatureLists(feature_list = feature_list)
        example = tf.train.SequenceExample(feature_lists = feature_lists)

        writer.write(example.SerializeToString())

    writer.close()
