import tensorflow as tf
import os
import glob
import cv2
import itertools
import random
import re

data_root = "C:/Projects/Cats-vs-Dogs/PetImages"
dirs = glob.glob(os.path.join(data_root, "*/"))

image_list = [glob.glob(os.path.join(indiv_dir, "*.jpg")) for indiv_dir in dirs]
images = list(itertools.chain(*image_list))

random.shuffle(images)

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

reg = re.compile(r".\\Cat\\\d+\.jpg$")

count = 1
    
for image in images:
    new_image = adjust_image(image)
    
    print(image)
    
    searchObj = re.search(reg, image)
    if searchObj:
        print("Cat iamge")
    else:
        print("Dog iamge")
    
    cv2.imshow("Resized image", new_image)
    assert (new_image.shape[1] == 320 and new_image.shape[0] == 240), "Invalid dimension"

    cv2.waitKey(0)
    
    count = count + 1
    if count == 10:
        cv2.destroyAllWindows()
        exit()

