import os
import glob
import cv2
import itertools
import random
import re
import numpy as np
import tensorflow as tf
import math

data_root = "C:\\Projects\\Cats-vs-Dogs\\PetImages"
save_path = "C:\\Data\\Cats-vs-Dogs"

dirs = glob.glob(os.path.join(data_root, "*\\"))

train_images = []
test_images = []

for indiv_dir in dirs:
    images = glob.glob(os.path.join(indiv_dir, "*.jpg"))
    pivot = int(len(images) * 0.8)
    train_images.extend(images[0:pivot])
    test_images.extend(images[pivot:])
    
random.shuffle(train_images)
random.shuffle(test_images)

# print(len(train_images))
# print(len(test_images))

HEIGHT=240
WIDTH=320
CHANNEL=3

ratio = HEIGHT / WIDTH

def crop_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    if img is None:
        raise Exception("Invalid JPEG file: {}".format(image_path))

    width = img.shape[1]
    height = img.shape[0]
    channel = img.shape[2]
    # print("width -> {}, height -> {}".format(width, height))

    if width < WIDTH or height < HEIGHT:
        raise Exception("JPEG file {} is too small".format(image_path))

    if channel != CHANNEL:
        raise Exception("Only processing color image file".format(image_path))

    resized = img[((height >> 1) - (HEIGHT >> 1)) : ((height >> 1) + (HEIGHT >> 1)),
                  ((width >> 1) - (WIDTH >> 1)) : ((width >> 1) + (WIDTH >> 1)), ...]

    return resized


def adjust_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    
    if img is None:
        raise Exception("Invalid JPEG file: {}".format(image_path))
        
    width = img.shape[1]
    height = img.shape[0]
    channel = img.shape[2]
    
    # print("width -> {}, height -> {}".format(width, height))
    
    if height / width >= ratio:
        scale = WIDTH / float(width)

        new_height = math.ceil(height * scale)
        dim = (WIDTH, new_height)
        
        if width >= WIDTH:
            resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        else:
            resized = cv2.resize(img, dim, interpolation = cv2.INTER_CUBIC)
        
        resized = resized[(int(new_height / 2) - (HEIGHT >> 1)) : (int(new_height / 2) + (HEIGHT >> 1)), ...]        
    else:
        scale = HEIGHT / float(height)
        
        new_width = int(width * scale)
        dim = (new_width, HEIGHT)
        
        if height >= HEIGHT:
            resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        else:
            resized = cv2.resize(img, dim, interpolation = cv2.INTER_CUBIC)
        
        resized = resized[:, (int(new_width / 2) - (WIDTH >> 1)) : (int(new_width / 2) + (WIDTH >> 1)), ...]        
        
    return resized

def float_feature(value):
    return tf.train.Feature(float_list = tf.train.FloatList(value = value))

def bytes_feature(value):
    return tf.train.Feature(bytes_list = tf.train.BytesList(value = [value]))

def int64_feature(value):
    return tf.train.Feature(int64_list = tf.train.Int64List(value = [value]))


def proc_record(image, reg, writer):
    print("processing iamge {}".format(image))
    try:
        new_image = crop_image(image)
        # new_image = adjust_image(image)
    except:
        return False
        
    assert (new_image.shape[1] == WIDTH and new_image.shape[0] == HEIGHT), "Invalid dimension"
        
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

    return True

reg = re.compile(r".\\Cat\\\d+\.jpg$")

print('beginning prepare PET tfrecords for training')
writer = tf.io.TFRecordWriter(os.path.join(save_path, 'pet-train.tfr'))
num_train_records = 0

for image in train_images:
    ret = proc_record(image, reg, writer)
    if ret: num_train_records = num_train_records + 1

writer.close()
print('end of tfrecords preparation for training')

print('beginning prepare PET tfrecords for testing')
writer = tf.io.TFRecordWriter(os.path.join(save_path, 'pet-test.tfr'))
num_test_records = 0

for image in test_images:
    ret = proc_record(image, reg, writer)
    if ret: num_test_records = num_test_records + 1

writer.close()
print('end of tfrecords preparation for testing')

print('#tfrecords for training: {}'.format(num_train_records))
print('#tfrecords for testing: {}'.format(num_test_records))
