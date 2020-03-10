from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import MaxPooling2D, AveragePooling2D
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Lambda, Flatten, Dense
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K

K.set_image_data_format("channels_first")

import cv2
import os
import numpy as np
from numpy import genfromtxt
import pandas as pd
import tensorflow as tf
from utils import *
import sys
from inception_blocks import *

np.set_printoptions(threshold=sys.maxsize)

FRmodel = faceRecoModel(input_shape=(3, 96, 96))
FRmodel.summary()


def triplet_loss(y_true, y_pred, alpha=0.2):
    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]
    pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), axis=-1)
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), axis=-1)
    basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), alpha)
    loss = tf.reduce_sum(tf.maximum(basic_loss, 0.0), axis=None)
    return loss

print("compiling...")
FRmodel.compile(optimizer='adam', loss=triplet_loss, metrics=['accuracy'])
print("load weights...")
load_weights_from_FaceNet(FRmodel)

# database = {}
# database["danielle"] = img_to_encoding("images/danielle.png", FRmodel)
# database["younes"] = img_to_encoding("images/younes.jpg", FRmodel)
# database["tian"] = img_to_encoding("images/tian.jpg", FRmodel)
# database["andrew"] = img_to_encoding("images/andrew.jpg", FRmodel)
# database["kian"] = img_to_encoding("images/kian.jpg", FRmodel)
# database["dan"] = img_to_encoding("images/dan.jpg", FRmodel)
# database["sebastiano"] = img_to_encoding("images/sebastiano.jpg", FRmodel)
# database["bertrand"] = img_to_encoding("images/bertrand.jpg", FRmodel)
# database["kevin"] = img_to_encoding("images/kevin.jpg", FRmodel)
# database["felix"] = img_to_encoding("images/felix.jpg", FRmodel)
# database["benoit"] = img_to_encoding("images/benoit.jpg", FRmodel)
# database["arnaud"] = img_to_encoding("images/arnaud.jpg", FRmodel)
# database["david"] = img_to_encoding("images/rsz_23836.jpg", FRmodel)

print("loading images...")
database = dict()
image_dir_path = "images"
images = os.listdir(image_dir_path)
for image in images:
    database[image.split(".")[0]] = [img_to_encoding(image_dir_path + "/" + image, FRmodel)]


def who_is_it(image_path, database, model):
    encoding = img_to_encoding(image_path, model)
    min_dist = 100
    for (name, db_enc) in database.items():
        dist = np.linalg.norm(encoding - db_enc)
        if dist < min_dist:
            min_dist = dist
            identity = name
    if min_dist > 0.7:
        print("Not in the database.")
    else:
        print("it's " + str(identity) + ", the distance is " + str(min_dist))
    return min_dist, identity

print("who_is_it")
who_is_it("rsz_david.png", database, FRmodel)
