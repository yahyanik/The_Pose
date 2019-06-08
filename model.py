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
from models import *

def imagelist (data1, coco_kps_f, imgIds_f, name):
    dataset = []
    new_labels = data1.labeling(coco_kps_f, imgIds_f, name)
    x = []
    y_t = []
    for i in range(0, len(imgIds_f)):

        img = coco_kps_f.loadImgs(imgIds_f[i])[0]
        imgFile = cv2.imread('../The_Pose/database/coco/images/'+name+'2017/' + img['file_name'])   #reading each image to put in the dataset variable
        if (i+1) % 10000 == 0:
            print ('10000 images are processed')
        # cv2.imshow('image1', imgFile)
        # cv2.waitKey(0)
        # labels = open("new_labels_" + name + ".txt", "r") #read the labels to put with each image
        # tmpIMG = imutils.resize(imgFile, width= 320, height= 320)
        tmpIMG = cv2.resize(imgFile, (320, 320))
        tmpLABEL = new_labels[imgIds_f[i]]
        dataset.append([tmpIMG, tmpLABEL])
        x.append(tmpIMG)
        y_t.append(tmpLABEL)  # label is a list object
        # y_t_numpy = np.array(y_t)
        # print (y_t_numpy.shape)
    X_t = np.array(x).reshape(-1, 320, 320, 3)
    y_t_numpy = np.array(y_t)
    Y_t = y_t_numpy.reshape([-1,10,10,26])
    pickle_out = open('../The_Pose/tfrecord/X_Corrected_{}_3.1.pickle'.format(name), 'wb')
    pickle.dump(X_t, pickle_out)
    pickle_out.close()

    pickle_out = open('../The_Pose/tfrecord/y_Corrected_{}_3.1.pickle'.format(name), 'wb')
    pickle.dump(Y_t, pickle_out)
    pickle_out.close()

    print (len(dataset), 'number of examples that we have for the', name)
    # return dataset


def Data_prepration ():
    data1 = data()
    dataset_training_ids, dataset_val_ids = data1.DataReshape()  #raeding dataset and preparing each image
    #training and val show the image lables
    coco_kps_t, imgIds_t = dataset_training_ids
    coco_kps_v, imgIds_v = dataset_val_ids
    write_tfrecord(data1, coco_kps_t, imgIds_t, 'train') # make the training dataset
    write_tfrecord(data1, coco_kps_v, imgIds_v, 'val')


    # x = []
    # y_t = []
    # for img, label in dataset_t:
    #     x.append(img)
    #     y_t.append(label)   #label is a list object
    # # y_t_numpy = np.array(y_t)
    # # print (y_t_numpy.shape)
    # X_t = np.array(x).reshape(-1, 320, 320, 3)
    # pickle_out = open('X_t.pickle' , 'wb')
    # pickle.dump (X_t, pickle_out)
    # pickle_out.close()
    #
    # pickle_out = open('y_t.pickle', 'wb')
    # pickle.dump(y_t, pickle_out)
    # pickle_out.close()

    imagelist(data1, coco_kps_v, imgIds_v, 'val')       # create the val dataset
    # x = []
    # y_v = []
    # for img, label in dataset_v:
    #     x.append(img)
    #     y_v.append(label)
    # X_v = np.array(x).reshape(-1, 320, 320, 3)
    # pickle_out = open('X_v.pickle', 'wb')
    # pickle.dump(X_v, pickle_out)
    # pickle_out.close()
    #
    # pickle_out = open('y_v.pickle', 'wb')
    # pickle.dump(y_v, pickle_out)
    # pickle_out.close()

    # return (dataset_t, dataset_v)




def model ():   # the model is defined here

    """Model function for CNN."""

    img_size = 320
    tfrecord_filename = '../The_Pose/tfrecord/keypoint_train.tfrecords'


    #creating the model
    model_mn2 = MobileNetV2(True, 320)
    print(model_mn2.output.get_shape())
    # board_writer = tf.summary.FileWriter(logdir='./', graph=tf.get_default_graph())

    # fake_data = np.ones(shape=(1, 320, 320, 3))

    # sess_config = tf.ConfigProto()
    # with tf.Session(config=sess_config) as sess:
        # sess.run(tf.global_variables_initializer())

        # cnt = 0
        # for i in range(101):
        #     t1 = time.time()
        #     output = sess.run(model.output, feed_dict={model.input: fake_data})
            # if i != 0:
                # cnt += time.time() - t1
        # print(cnt / 100)

    # model = Sequential()
    #
    # model.add(Conv2D(32, (3, 3), input_shape=X_t.shape[1:]))
    # model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    #
    # model.add(Conv2D(64, (3, 3), input_shape=X_t.shape[1:]))
    # model.add(Activation('relu'))
    #
    # model.add(Conv2D(64, (3, 3), input_shape=X_t.shape[1:]))
    # model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    #
    # model.add(Conv2D(128, (3, 3), input_shape=X_t.shape[1:]))
    # model.add(Activation('relu'))
    #
    # model.add(Conv2D(128, (3, 3), input_shape=X_t.shape[1:]))
    # model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    #
    # model.add(Conv2D(256, (3, 3), input_shape=X_t.shape[1:]))
    # model.add(Activation('relu'))
    #
    # model.add(Conv2D(256, (3, 3), input_shape=X_t.shape[1:]))
    # model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    #
    # model.add(Conv2D(256, (3, 3), input_shape=X_t.shape[1:]))
    # model.add(Activation('relu'))
    #
    # model.add(Conv2D(256, (3, 3), input_shape=X_t.shape[1:]))
    # model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    #
    # model.add(Conv2D(256, (3, 3), input_shape=X_t.shape[1:]))
    # model.add(Activation('relu'))
    #
    # model.add(Conv2D(256, (3, 3), input_shape=X_t.shape[1:]))
    # model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    #
    # model.add(Conv2D(256, (3, 3), input_shape=X_t.shape[1:]))
    # model.add(Activation('relu'))
    #
    # model.add(Conv2D(256, (3, 3), input_shape=X_t.shape[1:]))
    # model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    #
    # model.add(Conv2D(256, (3, 3), input_shape=X_t.shape[1:]))
    # model.add(Activation('relu'))
    #
    # model.add(Conv2D(256, (3, 3), input_shape=X_t.shape[1:]))
    # model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    #
    # model.add(Conv2D(512, (3, 3), input_shape=X_t.shape[1:]))
    # model.add(Activation('relu'))
    #
    # model.add(Conv2D(512, (3, 3), input_shape=X_t.shape[1:]))
    # model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    #
    # model.add(Conv2D(512, (3, 3), input_shape=X_t.shape[1:]))
    # model.add(Activation('relu'))
    #
    # model.add(Conv2D(512, (3, 3), input_shape=X_t.shape[1:]))
    # model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))

    return (X_t, Y_t, X_v, Y_v, model_mn2)


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def write_tfrecord (data1, coco_kps_f, imgIds_f, name):
    tfrecords_filename = '../The_Pose/tfrecord/keypoint_'+name+'_Corrected.1.tfrecords'
    writer = tf.python_io.TFRecordWriter(tfrecords_filename)

    new_labels = data1.labeling(coco_kps_f, imgIds_f, name)

    # original_images = []

    for i in range(0, len(imgIds_f)):
    # for i in range(0, 1):

        img = coco_kps_f.loadImgs(imgIds_f[i])[0]
        imgFile = cv2.imread('../The_Pose/database/coco/images/' + name + '2017/' + img['file_name'])  # reading each image to put in the dataset variable
        if (i + 1) % 10000 == 0:
            print('10000 images are processed')
        tmpIMG = cv2.resize(imgFile, (320, 320))
        annotation = new_labels[imgIds_f[i]]
        IMG = np.array(tmpIMG)
        # annotation = np.array(tmpLABEL)
        # original_images.append((img, annotation))
        height = IMG.shape[0]
        width = IMG.shape[1]

        img_raw = IMG.tostring()
        annotation_raw = annotation.tostring()

        example = tf.train.Example(features=tf.train.Features(feature={
            'height': _int64_feature(height),
            'width': _int64_feature(width),
            'image_raw': _bytes_feature(img_raw),
            # 'image_raw': _bytes_feature(IMG),
            'mask_raw': _bytes_feature(annotation_raw)}))
            # 'mask_raw': _bytes_feature(annotation)}))

        writer.write(example.SerializeToString())

    writer.close()

    # reconstructed_images = []

    # record_iterator = tf.python_io.tf_record_iterator(path=tfrecords_filename)
    #
    # for string_record in record_iterator:
    #     example = tf.train.Example()
    #     example.ParseFromString(string_record)
    #
    #     height = int(example.features.feature['height']
    #                  .int64_list
    #                  .value[0])
    #
    #     width = int(example.features.feature['width']
    #                 .int64_list
    #                 .value[0])
    #
    #     img_string = (example.features.feature['image_raw']
    #         .bytes_list
    #         .value[0])
    #
    #     annotation_string = (example.features.feature['mask_raw']
    #         .bytes_list
    #         .value[0])
    #
    #     img_1d = np.fromstring(img_string, dtype=np.uint8)
    #     reconstructed_img = img_1d.reshape((height, width, -1))
    #
    #     annotation_1d = np.fromstring(annotation_string, dtype=np.float16)
    #
    #     # Annotations don't have depth (3rd dimension)
    #     reconstructed_annotation = annotation_1d.reshape((10, 10, 26))

        # reconstructed_images.append((reconstructed_img, reconstructed_annotation))

    print( 'number of examples that we have for the', name)
    # return dataset

    # for original_pair, reconstructed_pair in zip(original_images, reconstructed_images):
    #     img_pair_to_compare, annotation_pair_to_compare = zip(original_pair,
    #                                                           reconstructed_pair)
    #     print(np.array_equal(img_pair_to_compare[0],img_pair_to_compare[1]))
    #     print(np.array_equal(annotation_pair_to_compare[0],annotation_pair_to_compare[1]))
    #
    #     cv2.imshow('image1', reconstructed_img)
    #
    #     cv2.imshow('image2', IMG)
    #     cv2.waitKey(0)







if __name__ == '__main__':


    Data_prepration()

    # X_t, Y_t, X_v, Y_v, model_mn2 = model()

