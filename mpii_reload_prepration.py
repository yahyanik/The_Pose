from __future__ import division
import pandas as pd
from scipy.io import loadmat
import cv2
import numpy as np
import tensorflow as tf
import pickle
from training import data


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def labeling(person,y, x):

    label = np.zeros((10, 10, 26), dtype=np.float16)  # placeholder for the labels in the annotations

    print (person.shape)
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
            print('data is saved successfully')

        return label, flag
    except:
        print('ERROR in reading the image')
        flag = False
        new_labels = 0
        return new_labels, flag

    # f = open("new_labels_"+filename+".txt","w")



# x = []
# x_val = []
# y_t = []
# y_val = []
tfrecords_filename = '../The_Pose/tfrecord/keypoint_train_MPII.11.tfrecords'
writer = tf.python_io.TFRecordWriter(tfrecords_filename)

data = loadmat('../The_Pose/database/mpii_human_pose_v1_u12_2/mpii_human_pose_v1_u12_1.mat')
anno = data['RELEASE']['annolist'][0,0]

for i, CurrentImg in enumerate(anno[0,:]):
    print (i)
    imgFile = cv2.imread('../The_Pose/database/images/'+CurrentImg['image']['name'][0][0][0])
    print('../The_Pose/database/images/'+CurrentImg['image']['name'][0][0][0])
    if (i + 1) % 10000 == 0:
        print('10000 images are processed')

    try:
        height, width, _ = imgFile.shape
        tmpLABEL, flag = labeling(CurrentImg['annorect'], height, width)

    except:
        continue
    print (flag)
    if not flag:
        continue

    tmpIMG = cv2.resize(imgFile, (320, 320))
    IMG = np.array(tmpIMG)
    annotation = tmpLABEL

    # cv2.circle(imgFile, (key[0],key[1]), 2, (0, 0, 255), -1)
    # cv2.circle(imgFile, (key[15],key[16]), 2, (0, 0, 255), -1)
    # cv2.circle(imgFile, (key[18],key[19]), 2, (0, 0, 255), -1)
    # cv2.circle(imgFile, (key[21],key[22]), 2, (0, 0, 255), -1)
    # cv2.circle(imgFile, (key[24],key[25]), 2, (0, 0, 255), -1)
    # cv2.circle(imgFile, (key[27], key[28]), 2, (0, 0, 255), -1)
    # cv2.circle(imgFile, (key[30], key[31]), 2, (0, 0, 255), -1)
    # cv2.circle(imgFile, (int(center[0]), int(center[1])),4,(255,0,200),-1)
    # cv2.rectangle(imgFile, (int(bbox[0]), int(bbox[1])), (int(bbox[0]+bbox[2]), int(bbox[1]+bbox[3])), (0,255,0),1)
    # x = 320
    # y = 320
    # cv2.line(tmpIMG, (int(x/10), 0), (int(x/10), y), (255, 0, 0), 1)
    # cv2.line(tmpIMG, (int(2*x / 10), 0), (int(2*x / 10), y), (255, 0, 0), 1)
    # cv2.line(tmpIMG, (int(3 * x / 10), 0), (int(3 * x / 10), y), (255, 0, 0), 1)
    # cv2.line(tmpIMG, (int(4 * x / 10), 0), (int(4 * x / 10), y), (255, 0, 0), 1)
    # cv2.line(tmpIMG, (int(5 * x / 10), 0), (int(5 * x / 10), y), (255, 0, 0), 1)
    # cv2.line(tmpIMG, (int(6 * x / 10), 0), (int(6 * x / 10), y), (255, 0, 0), 1)
    # cv2.line(tmpIMG, (int(7 * x / 10), 0), (int(7 * x / 10), y), (255, 0, 0), 1)
    # cv2.line(tmpIMG, (int(8 * x / 10), 0), (int(8 * x / 10), y), (255, 0, 0), 1)
    # cv2.line(tmpIMG, (int(9 * x / 10), 0), (int(9 * x / 10), y), (255, 0, 0), 1)
    #
    # cv2.line(tmpIMG, (0, int(y/10)), (x, int(y/10)), (255, 0, 0), 1)
    # cv2.line(tmpIMG, (0, int(2*y / 10)), (x, int(2*y / 10)), (255, 0, 0), 1)
    # cv2.line(tmpIMG, (0, int(3*y / 10)), (x, int(3*y / 10)), (255, 0, 0), 1)
    # cv2.line(tmpIMG, (0, int(4*y / 10)), (x, int(4*y / 10)), (255, 0, 0), 1)
    # cv2.line(tmpIMG, (0, int(5*y / 10)), (x, int(5*y / 10)), (255, 0, 0), 1)
    # cv2.line(tmpIMG, (0, int(6*y / 10)), (x, int(6*y / 10)), (255, 0, 0), 1)
    # cv2.line(tmpIMG, (0, int(7*y / 10)), (x, int(7*y / 10)), (255, 0, 0), 1)
    # cv2.line(tmpIMG, (0, int(8*y / 10)), (x, int(8*y / 10)), (255, 0, 0), 1)
    # cv2.line(tmpIMG, (0, int(9*y / 10)), (x, int(9*y / 10)), (255, 0, 0), 1)
    #
    # cv2.imshow('image1', tmpIMG)
    # cv2.waitKey(0)
    # print (annotation)

    #array([0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    ,
       # 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    ,
       # 0.    , 0.    , 0.    , 0.    , 0.    , 1.    , 0.6406, 0.5693,
       # 4.72  , 2.797 ], dtype=float16)

    img_raw = IMG.tostring()
    annotation_raw = annotation.tostring()
    height1 = IMG.shape[0]
    width1 = IMG.shape[1]

    if data['RELEASE']['img_train'][0][0][0][i]:
        # x.append(tmpIMG)
        # y_t.append(tmpLABEL)
        example = tf.train.Example(features=tf.train.Features(feature={
            'height': _int64_feature(height1),
            'width': _int64_feature(width1),
            'image_raw': _bytes_feature(img_raw),
            # 'image_raw': _bytes_feature(IMG),
            'mask_raw': _bytes_feature(annotation_raw)}))
        writer.write(example.SerializeToString())

    # print(len(x))
    # example = tf.train.Example(features=tf.train.Features(feature={
    #     'height': _int64_feature(height),
    #     'width': _int64_feature(width),
    #     'image_raw': _bytes_feature(img_raw),
    #     # 'image_raw': _bytes_feature(IMG),
    #     'mask_raw': _bytes_feature(annotation_raw)}))
    # writer.write(example.SerializeToString())
writer.close()

# X_val = np.array(x_val).reshape(-1, 320, 320, 3)
# X_t = np.array(x).reshape(-1, 320, 320, 3)
#
#
# pickle_out = open('../The_Pose/tfrecord/X_train_mpii_3.pickle', 'wb')
# pickle.dump(X_t, pickle_out)
# pickle_out.close()
# pickle_out = open('../The_Pose/tfrecord/y_train_mpii_3.pickle', 'wb')
# pickle.dump(y_t, pickle_out)
# pickle_out.close()
# print(len(x), 'number of examples that we have for the', 'train')
#
# pickle_out = open('../The_Pose/tfrecord/X_test_mpii_3.pickle', 'wb')
# pickle.dump(X_val, pickle_out)
# pickle_out.close()
# pickle_out = open('../The_Pose/tfrecord/y_test_mpii_3.pickle', 'wb')
# pickle.dump(y_val, pickle_out)
# pickle_out.close()
# print(len(x_val), 'number of examples that we have for the', 'test')


