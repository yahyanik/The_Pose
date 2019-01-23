from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from training import data


import numpy as np
import tensorflow as tf

import cv2
import imutils

def Data_prepration (batch_size):
    data = data()
    training, val, dataset_training, dataset_val = data.DataReshape()
    coco_kps_t, imgIds_t = dataset_training
    coco_kps_v, imgIds_v = dataset_val
    dataset = []
    for i in range(0, len(imgIds_t)):
        batch = []
        for ii in range (0,batch_size):
            try:
                img = coco_kps_t.loadImgs(imgIds_t[i])[0]
                imgFile = cv2.imread('../The_Pose/database/coco/images/val2017/' + img['file_name'])
                batch.append(imutils.resize(imgFile, 320, 320))
            except:
                break
        batch_np = np.array(batch)
        dataset.append(batch_np)
    dataset_np = np.array(dataset)






def model (features, labels, mode):
    """Model function for CNN."""
    # Input Layer

    input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])

