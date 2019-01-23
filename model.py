from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from training import data


import numpy as np
import tensorflow as tf

import cv2
import imutils

def Data_prepration ():
    data1 = data()
    training, val, dataset_training, dataset_val = data1.DataReshape()
    coco_kps_t, imgIds_t = dataset_training
    coco_kps_v, imgIds_v = dataset_val
    print ('imgeID', imgIds_t [0])
    images_t = imagelist(coco_kps_t, imgIds_t)
    images_v = imagelist(coco_kps_v, imgIds_v)

    return (images_t, images_v)


def imagelist (coco_kps_t, imgIds_t):
    dataset = []
    for i in range(0, len(imgIds_t)):

        img = coco_kps_t.loadImgs(imgIds_t[i])[0]
        imgFile = cv2.imread('../The_Pose/database/coco/images/val2017/' + img['file_name'])
        cv2.imshow('image1', imgFile)
        cv2.waitKey(0)
        dataset.append(imutils.resize(imgFile, width = 320, height = 320))

    print ('the data is put in the list')
    return dataset






def model ():
    """Model function for CNN."""
    # Input Layer

    # input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])
    images_t, images_v = Data_prepration()
    labels_t = open("new_labels_"+'train'+".txt","r")
    labels_v = open("new_labels_"+'val'+".txt","r")



model()
