from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D

import os
import matplotlib.pyplot as plt
import imutils
import random
import pickle
import numpy as np
import cv2
from training import data

def Data_prepration ():
    data1 = data()
    training, val, dataset_training_ids, dataset_val_ids = data1.DataReshape()  #raeding dataset and preparing each image
    coco_kps_t, imgIds_t = dataset_training_ids
    coco_kps_v, imgIds_v = dataset_val_ids
    dataset_t = imagelist(coco_kps_t, imgIds_t, 'train')
    dataset_v = imagelist(coco_kps_v, imgIds_v, 'val')

    x = []
    y_t = []
    for img, label in dataset_t:
        x.append(img)
        y_t.append(label)
    X_t = np.array(x).reshape(-1, 320, 320, 1)
    pickle_out = open('X_t.pickle' , 'wb')
    pickle.dump (X_t, pickle_out)
    pickle_out.close()

    pickle_out = open('y_t.pickle', 'wb')
    pickle.dump(y_t, pickle_out)
    pickle_out.close()

    x = []
    y_v = []
    for img, label in dataset_v:
        x.append(img)
        y_v.append(label)
    X_v = np.array(x).reshape(-1, 320, 320, 1)
    pickle_out = open('X_v.pickle', 'wb')
    pickle.dump(X_v, pickle_out)
    pickle_out.close()

    pickle_out = open('y_v.pickle', 'wb')
    pickle.dump(y_v, pickle_out)
    pickle_out.close()

    # return (dataset_t, dataset_v)


def imagelist (coco_kps_f, imgIds_f, name):
    dataset = []
    for i in range(0, len(imgIds_f)):

        img = coco_kps_f.loadImgs(imgIds_f[i])[0]
        imgFile = cv2.imread('../The_Pose/database/coco/images/'+name+'2017/' + img['file_name'])   #reading each image to put in the dataset variable
        if i % 10000 == 0:
            print ('10000 images are processed')
        # cv2.imshow('image1', imgFile)
        # cv2.waitKey(0)
        labels = open("new_labels_" + name + ".txt", "r") #read the labels to put with each image

        dataset.append([imutils.resize(imgFile, width= 320, height= 320), labels[imgIds_f[i]]])

    print (len(dataset), 'number of examples that we have for the', name)
    return dataset






def model ():   # the model is defined here
    """Model function for CNN."""
    # making the dataset ready for training
    Data_prepration()    #prepare the data and write to the file system so the next time we will just read the dataset


    # reading the dataset from the file system
    X_t = pickle.load(open('X_t.pickle', 'rb'))
    y_t = pickle.load(open('y_t.pickle', 'rb'))
    X_v = pickle.load(open('X_v.pickle', 'rb'))
    y_v = pickle.load(open('y_v.pickle', 'rb'))

    # normalizing the data


    #creating the model

    model = Sequential()
    model.add(Conv2D(256, (3, 3), input_shape=X.shape[1:]))
    model.add(Activation('relu'))




    # random.shuffle(dataset_t)











    # labels_t = open("new_labels_"+'train'+".txt","r")
    # labels_v = open("new_labels_"+'val'+".txt","r")





model()
