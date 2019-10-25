from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from scipy.io import loadmat

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


def labeling_mpii(person,y, x):

    label = np.zeros((10, 10, 26), dtype=np.float16)  # placeholder for the labels in the annotations

    # print (person.shape)
    try:
        for i in range(person.shape[1]):


            try:
                Scale = person['scale'][0][i][0][0]
                PositionX = (person['objpos'][0,i][0][0][0][0][0]/x)*10
                PositionY = (person['objpos'][0,i][0][0][1][0][0]/y)*10

            except:
                flag = False
                return label,flag

            id_pos_y = int(PositionY)
            id_pos_x = int(PositionX)

            pos_x_new = PositionX - id_pos_x
            pos_y_new = PositionY - id_pos_y

            obj_Hight = Scale * 200
            obj_Weight = obj_Hight / 3
            x1 = obj_Hight / (x / 10)
            x2 = obj_Weight / (y / 10)

            label[int(id_pos_x), int(id_pos_y), 21] = 1  # pc =1
            label[int(id_pos_x), int(id_pos_y), 22] = pos_x_new  # center
            label[int(id_pos_x), int(id_pos_y), 23] = pos_y_new
            label[int(id_pos_x), int(id_pos_y), 24] = x1  # bounding size
            label[int(id_pos_x), int(id_pos_y), 25] = x2

            try:
                # id = []
                # x_keypoint = []
                # y_keypoint = []
                xx = 0
                yy = 0
                for j in (person['annopoints'][0][i]['point'][0][0][0]):   #print (person['annopoints'][0,0]['point'][0][0][0][0][0])
                    JointId = (j[2][0][0])
                    x_keypoint = (j[0][0][0]/x)*10
                    y_keypoint = ((j[1][0][0])/y)*10 # putting the labels in the new 10*10 window dimention

                    id_x = int(x_keypoint)
                    id_y = int(y_keypoint)
                    x_keypoint_new = x_keypoint - id_x
                    y_keypoint_new = y_keypoint - id_y


                    if JointId == 8:
                        xx = j[0][0][0]
                        yy = j[1][0][0]

                    if JointId == 9:  # putting data in their respected palce and change label to show what part is detected
                        if xx!=0 and yy!=0:
                            x_keypoint = (((j[0][0][0]+xx)/2)/x)*10
                            y_keypoint = (((j[1][0][0]+yy)/2)/y)*10
                        else:
                            x_keypoint = x_keypoint_new
                            y_keypoint = y_keypoint_new
                        id_x = int(x_keypoint)
                        id_y = int(y_keypoint)
                        x_keypoint_new = x_keypoint - id_x
                        y_keypoint_new = y_keypoint - id_y

                    if JointId == 9:
                        label[int(id_x), int(id_y), 0] = x_keypoint_new
                        label[int(id_x), int(id_y), 1] = y_keypoint_new
                        label[int(id_x), int(id_y), 2] = 1  # 6 or 11 for left sholder
                    if JointId == 13:
                        label[int(id_x), int(id_y), 3] = x_keypoint_new
                        label[int(id_x), int(id_y), 4] = y_keypoint_new
                        label[int(id_x), int(id_y), 5] = 1  # 6 or 11 for left sholder
                    if JointId == 12:
                        label[int(id_x), int(id_y), 6] = x_keypoint_new
                        label[int(id_x), int(id_y), 7] = y_keypoint_new
                        label[int(id_x), int(id_y), 8] = 1 # 7 or 12 for right sholder
                    if JointId == 14:
                        label[int(id_x), int(id_y), 9] = x_keypoint_new
                        label[int(id_x), int(id_y), 10] = y_keypoint_new
                        label[int(id_x), int(id_y), 11] = 1 # 8 or 13 for left arm
                    if JointId == 11:
                        label[int(id_x), int(id_y), 12] = x_keypoint_new
                        label[int(id_x), int(id_y), 13] = y_keypoint_new
                        label[int(id_x), int(id_y), 14] = 1 # 9 or 14 for right arm
                    if JointId == 15:
                        label[int(id_x), int(id_y), 15] = x_keypoint_new
                        label[int(id_x), int(id_y), 16] = y_keypoint_new
                        label[int(id_x), int(id_y), 17] =1  # 9 or 14 for left hand
                    if JointId == 10:
                        label[int(id_x), int(id_y), 18] = x_keypoint_new
                        label[int(id_x), int(id_y), 19] = y_keypoint_new
                        label[int(id_x), int(id_y), 20] = 1 # 9 or 14 for right hand
            except:
                flag = False
                return label, flag


            flag = True

            if (i + 1) % 10000 == 0:
                print('10000 images are recorded')
            # print('data is saved successfully')

        return label, flag
    except:
        print('ERROR in reading the image')
        flag = False
        new_labels = 0
        return new_labels, flag

    # f = open("new_labels_"+filename+".txt","w")


def write_tfrecord_val(data1, coco_kps_f, imgIds_f, name):
    tfrecords_filename = '../The_Pose/tfrecord/DATASET_BLUR_VAL.tfrecords'
    writer = tf.python_io.TFRecordWriter(tfrecords_filename)
    new_labels = data1.labeling(coco_kps_f, imgIds_f, name)
    num = 0
    for i in range(0, len(imgIds_f)):
        img = coco_kps_f.loadImgs(imgIds_f[i])[0]
        imgFile = cv2.imread('/media/yahya/9EEA399CEA39721F/Users/yahya/Desktop/__DATASET__/database/coco/images/' + name + '2017/' + img[
            'file_name'])  # reading each image to put in the dataset variable
        if (i + 1) % 10000 == 0:
            print('10000 images are processed')
        num+=1
        tmpIMG = cv2.resize(imgFile, (320, 320))
        tmpIMG = cv2.GaussianBlur(tmpIMG, (5, 5), 0)
        annotation = new_labels[imgIds_f[i]]
        IMG = np.array(tmpIMG)
        height = IMG.shape[0]
        width = IMG.shape[1]
        img_raw = IMG.tostring()
        annotation_raw = annotation.tostring()
        example = tf.train.Example(features=tf.train.Features(feature={
            'height': _int64_feature(height),
            'width': _int64_feature(width),
            'image_raw': _bytes_feature(img_raw),
            'mask_raw': _bytes_feature(annotation_raw)}))
        writer.write(example.SerializeToString())
    writer.close()
    print('number of examples that we have for the val', num)


def Data_prepration ():
    data1 = data()
    dataset_training_ids, dataset_val_ids = data1.DataReshape()  #raeding dataset and preparing each image
    coco_kps_t, imgIds_t = dataset_training_ids
    coco_kps_v, imgIds_v = dataset_val_ids
    # write_tfrecord(data1, coco_kps_t, imgIds_t, 'train') # make the training dataset
    write_tfrecord_val(data1, coco_kps_v, imgIds_v, 'val')  # make the training dataset       #for the training data


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def write_tfrecord (data1, coco_kps_f, imgIds_f, name):
    tfrecords_filename = '../The_Pose/tfrecord/DATASET_BLUR_COCO.tfrecords'
    writer = tf.python_io.TFRecordWriter(tfrecords_filename)

    new_labels = data1.labeling(coco_kps_f, imgIds_f, name)
    num = 0

    for i in range(0, len(imgIds_f)):

        img = coco_kps_f.loadImgs(imgIds_f[i])[0]
        imgFile = cv2.imread('/media/yahya/9EEA399CEA39721F/Users/yahya/Desktop/__DATASET__/database/coco/images/' + name + '2017/' + img['file_name'])  # reading each image to put in the dataset variable
        if (i) % 100 == 0:
            print('100 images are processed',i)
        num+=1
        tmpIMG = cv2.resize(imgFile, (320, 320))
        tmpIMG = cv2.GaussianBlur(tmpIMG, (5, 5), 0)
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

    # data = loadmat('/media/yahya/9EEA399CEA39721F/Users/yahya/Desktop/__DATASET__/database/mpii_human_pose_v1_u12_2/mpii_human_pose_v1_u12_1.mat')
    # anno = data['RELEASE']['annolist'][0, 0]
    #
    # for i, CurrentImg in enumerate(anno[0, :]):
    #     # print(i)
    #     imgFile = cv2.imread('/media/yahya/9EEA399CEA39721F/Users/yahya/Desktop/__DATASET__/database/images/' + CurrentImg['image']['name'][0][0][0])
    #     # print('/media/yahya/9EEA399CEA39721F/Users/yahya/Desktop/__DATASET__/database/images/' + CurrentImg['image']['name'][0][0][0])
    #     if (i + 1) % 10000 == 0:
    #         print('10000 images are processed')
    #
    #     try:
    #         height, width, _ = imgFile.shape
    #         tmpLABEL, flag = labeling_mpii(CurrentImg['annorect'], height, width)
    #
    #     except:
    #         continue
    #     # print(flag)
    #     if not flag:
    #         continue
    #
    #     tmpIMG = cv2.resize(imgFile, (320, 320))
    #     IMG = np.array(tmpIMG)
    #     annotation = tmpLABEL
    #     img_raw = IMG.tostring()
    #     annotation_raw = annotation.tostring()
    #     height1 = IMG.shape[0]
    #     width1 = IMG.shape[1]
    #
    #     if data['RELEASE']['img_train'][0][0][0][i]:
    #         # x.append(tmpIMG)
    #         # y_t.append(tmpLABEL)
    #         num+=1
    #         example = tf.train.Example(features=tf.train.Features(feature={
    #             'height': _int64_feature(height1),
    #             'width': _int64_feature(width1),
    #             'image_raw': _bytes_feature(img_raw),
    #             # 'image_raw': _bytes_feature(IMG),
    #             'mask_raw': _bytes_feature(annotation_raw)}))
    #         writer.write(example.SerializeToString())

    writer.close()

    print( 'number of examples that we have for the train', num)








if __name__ == '__main__':


    Data_prepration()


