import tensorflow as tf
import os
import glob
import cv2
import itertools
import random
import re

data_root = "C:/Projects/Cats-vs-Dogs/PetImages"
save_path = 'data/pet'

dirs = glob.glob(os.path.join(data_root, "*/"))

image_list = [glob.glob(os.path.join(indiv_dir, "*.jpg")) for indiv_dir in dirs]
images = list(itertools.chain(*image_list))

random.shuffle(images)

pivot = len(images) * 0.8

train_images = images[0:pivot]
test_images = images[(pivot+1):]

ratio = 240 / 320

def adjust_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    
    width = img.shape[1]
    height = img.shape[0]
    
    # print("width -> {}, height -> {}".format(width, height))
    
    if height / width >= ratio:
        scale = 320 / float(width)

        new_height = int(height * scale)
        dim = (320, new_height)
        
        if width >= 320:
            resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        else:
            resized = cv2.resize(img, dim, interpolation = cv2.INTER_CUBIC)
        
        resized = resized[(int(new_height / 2) - 120) : (int(new_height / 2) + 120), ...]        
    else:
        scale = 240 / float(height)
        
        new_width = int(width * scale)
        dim = (new_width, 240)
        
        if height >= 240:
            resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        else:
            resized = cv2.resize(img, dim, interpolation = cv2.INTER_CUBIC)
        
        resized = resized[:, (int(new_width / 2) - 160) : (int(new_width / 2) + 160), ...]        
        
    return resized

def float_feature(value):
    return tf.train.Feature(float_list = tf.train.FloatList(value = value))

def bytes_feature(value):
    return tf.train.Feature(bytes_list = tf.train.BytesList(value = [value]))

def int64_feature(value):
    return tf.train.Feature(int64_list = tf.train.Int64List(value = [value]))


def proc_record(image, reg, writer):
    new_image = adjust_image(image)
    assert (new_image.shape[1] == 320 and new_image.shape[0] == 240), "Invalid dimension"
        
    searchObj = re.search(reg, image)
    if searchObj:
        # print("Cat iamge")
        label = 0
    else:
        # print("Dog iamge")
        label = 1

    # cv2.imshow("Resized image", new_image)
    # cv2.waitKey(0)    
    # cv2.destroyAllWindows()

    new_image = new_image / 255.0
    
    feature = \
    {
        'image': float_feature(np.reshape(new_image, [-1])),
        'label': int64_feature(label)
    }
    example = tf.train.Example(features = tf.train.Features(feature = feature))
    writer.write(example.SerializeToString())


reg = re.compile(r".\\Cat\\\d+\.jpg$")

print('beginning prepare PET tfrecords for training')
writer = tf.io.TFRecordWriter(os.path.join(save_path, 'pet-train.tfr'))
num_train_records = 0

for image in train_images:
    proc_record(image, reg, writer)
    num_train_records = num_train_records + 1

writer.close()
print('end of tfrecords preparation for training')

print('beginning prepare PET tfrecords for testing')
writer = tf.io.TFRecordWriter(os.path.join(save_path, 'pet-test.tfr'))
num_test_records = 0

for image in test_images:
    proc_record(image, reg, writer)
    num_test_records = num_test_records + 1

writer.close()
print('end of tfrecords preparation for testing')

print('#tfrecords for training: {}'.format(num_train_records))
print('#tfrecords for testing: {}'.format(num_test_records))
