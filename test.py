
# from __future__ import division
from pycocotools.coco import COCO
import numpy as np
import matplotlib.pyplot as plt
import pylab
import cv2
import pickle
import tensorflow.contrib as tc
import tensorflow as tf
from Metrics1 import *
from tensorflow import keras
import skimage.io as io


img1 = cv2.imread('000061164.jpg')
img = cv2.GaussianBlur(img1,(5,5),0)
img = cv2.resize(img, (320,320))

img2 = cv2.resize(img1, (320,320))
img2 = cv2.GaussianBlur(img2,(5,5),0)
cv2.imshow('img', img)
cv2.imshow('img1', img1)
cv2.imshow( 'img2', img2)
cv2.waitKey(0)

# for i in range (10):
#     print (i)
# with open('../The_Pose/tfrecord/X_new_val_3.pickle', 'rb') as f:
#     X_v = pickle.load(f)
# X_val = np.array(X_v)
# with open('../The_Pose/tfrecord/y_new_val_3.pickle', 'rb') as f:
#     y_v = pickle.load(f)
# y_val = np.array(y_v)
#
# print (X_val.shape)
# x = X_val[1*32:(1+1)*32,:,:,:]
# print (x.shape)
# cv2.imshow('tracking', x[0,:,:,:])
#
# cv2.waitKey(0)

#
# pylab.rcParams['figure.figsize'] = (8.0, 10.0)
# dataDir='..'
# dataType='val2017'
# coco=COCO('../The_Pose/database/coco/annotations/instances_train2017.json'.format(dataDir,dataType))
# catIds = coco.getCatIds(catNms=['person'])
#
#
# catIds = coco.getCatIds(catNms=['person']);
# imgIds = coco.getImgIds(catIds=catIds );
# img = coco.loadImgs(imgIds[6])[0]
#
# I = io.imread(img['coco_url'])
# plt.axis('off')
# plt.imshow(I)
# plt.show()
#
# plt.imshow(I); plt.axis('off')
# annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
# anns = coco.loadAnns(annIds)
# coco.showAnns(anns)
# annFile = '../The_Pose/database/coco/annotations/person_keypoints_train2017.json'.format(dataDir,dataType)
# coco_kps=COCO(annFile)
# plt.imshow(I); plt.axis('off')
# ax = plt.gca()
# annIds = coco_kps.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
# anns = coco_kps.loadAnns(annIds)
# coco_kps.showAnns(anns)
# print 'kk'

# img = cv2.imread ('../The_Pose/database/coco/images/val2017/000000532481.jpg')
# print img
# cv2.imshow('img', img)
# cv2.waitKey(0)
# np_w = np.array ([[[[0],[6]],[[2],[7]],[[3],[8]],[[4],[9]]],[[[12],[30]],[[40],[13]],[[50],[14]],[[60],[15]]], [[[31],[22]],[[41],[23]],[[51],[24]],[[61],[16]]]])
#
# tfrecords_filename_mpii = '../The_Pose/tfrecord/keypoint_train_DA.tfrecords'
# filename_queue_mpii = tf.train.string_input_producer([tfrecords_filename_mpii])
# image_mpii, annotation_mpii = read_and_decode(320, filename_queue_mpii, 16, 10)
#
# a1 = tf.Variable ([[1,30,0,0],[1,0,1.5,0],[1,0,1,20]])
# #
# a = tf.Variable ([[1,0,1,0],[1.1,0,0,1],[0,1,1,0]])
#
# init_op = tf.initialize_all_variables()
# with tf.Session() as sess:
#
#
# #     # Run the variable initializer.
#     sess.run(init_op)
#     sess.run(tf.global_variables_initializer())
#     sess.run(tf.local_variables_initializer())
#     print (sess.run(tf.reduce_max(a1)))
#     coord = tf.train.Coordinator()
#     threads = tf.train.start_queue_runners(coord=coord)
# # #     # print sess.run(a[:,3:4])
#     img_mpii, anno_mpii = sess.run([image_mpii, annotation_mpii])
#     print (img_mpii.shape)

# print (sess.run(a1[:,2:]))









#     print (sess.run(tf.add_n([a,a1])))
#     print (sess.run(tf.reduce_sum(a,a1, axis = -1)))
#     print sess.run(tf.squared_difference(a , a1))
#     print sess.run(tf.reduce_sum(tf.squared_difference(a , a1)))

#     k = (sess.run(tf.concat([a, a1], 1)))
#     print k
#     print k.shape
#     print (sess.run(tf.to_int32(tf.round(ww))))
#     ff = sess.run(tf.cast(tf.equal(tf.to_int32(tf.round(ww)), tf.to_int32(tf.round(www))) , 'float'))
#
#     print (ff)
#     print (sess.run(tf.reduce_mean(ff)))
#     print ('this is with flatten', (sess.run(yy)))
#     print (yy.shape)
#     print (sess.run(tf.sign(y)))
#     print (sess.run(y*tf.sign(y)))
#     print (sess.run(5*y[1]*yy[0,2]))
#     print (sess.run(keras.backend.mean(tf.keras.backend.square(w-w))))
    # print ('this is with reshape', (sess.run(y)))




# np_w = np.array ([[[[1],[6]],[[2],[7]],[[3],[8]],[[4],[9]]],[[[12],[30]],[[40],[13]],[[50],[14]],[[60],[15]]], [[[31],[22]],[[41],[23]],[[51],[24]],[[61],[16]]]])
# np_y = np_w.reshape([np_w.shape[0],-1])
# print np_y
# np_y = np_y.reshape([np_w.shape[0],np_w.shape[1],np_w.shape[2],np_w.shape[3]])
# print np_y

'''
dataDir='..'
dataType='train2017'
coco_kps=COCO('../The_Pose/database/coco/annotations/person_keypoints_train2017.json'.format(dataDir, dataType))
imgIds = coco_kps.getImgIds(catIds=coco_kps.getCatIds(catNms=['person']))
for i in range (116,117):
            img = coco_kps.loadImgs(imgIds[i])[0]
            imgFile = cv2.imread('../The_Pose/database/coco/images/' + "train" + '2017/' + img[
                'file_name'])  # reading each image to put in the dataset variable

            x = img['width']
            y = img['height']
            annIds = coco_kps.getAnnIds(imgIds=imgIds[i])
            label = np.zeros((10,10,15))
            print annIds
            imgFile = cv2.resize(imgFile, (320, 320))
            cv2.imshow('img', imgFile)

            cv2.waitKey(0)
            for id in annIds:
                anns = coco_kps.loadAnns(id)
                print anns

for i in range (11,12):
    print i
'''


# a = np.array([[0.1,0.2,0.3],[0.4,0.5,0.6]] , dtype=np.float16)
# print a
# print a.shape
# aa = a.tostring()
# print len(aa)
# print aa
#
# a_1d = np.fromstring(aa, dtype=np.uint8)
# kk = a_1d.reshape((2,3))
#
# print kk
# print kk.shape



# y_true = np.array([[2], [1], [0], [3], [0]]).astype(np.int64)
# y_true = tf.identity(y_true)
#
# y_pred = np.array([[0.0],
#                    [0.8],
#                    [0.3],
#                    [0.0],
#                    [0.1]
#                    ]).astype(np.float32)
# y_pred = tf.identity(y_pred)
# print y_pred.shape
#
# _, m_ap = tf.metrics.sparse_average_precision_at_k(y_true, y_pred, 1)
#
# sess = tf.Session()
# sess.run(tf.local_variables_initializer())
#
# tf_map = sess.run(tf.reduce_sum(y_pred))
# count = sess.run(tf.count_nonzero(y_pred))
# print(tf_map/count)
#
# print (sess.run(tf.sqrt(y_pred)))
#
# for i in range(5):
#     print i
# a1 = tf.Variable ([[1,0,0,0],[1,0,1,0],[1,0,1,0]])
#
# a = tf.Variable ([[1,0,1,0],[1,0,0,1],[0,1,1,0]])
# # b = tf.Variable ([[-11],[0],[-3]])
#
# # c = tf.cond(0 < a < 4, lambda: tf.assign(a,1), lambda: tf.assign(a,0))
# # d = a/b
#
# sess = tf.Session()
# sess.run(tf.global_variables_initializer())
# sess.run(tf.local_variables_initializer())
# # print sess.run(tf.reduce_sum(a))
# # print sess.run(tf.count_nonzero(a))
# print (a.shape)
# print sess.run(tf.metrics.true_positives(a1,a))

# tfrecords_filename = '../The_Pose/tfrecord/keypoint_train_new.tfrecords'
# filename_queue = tf.train.string_input_producer([tfrecords_filename])
# with open('../The_Pose/tfrecord/X_new_val.pickle', 'rb') as f:
#     X_v = pickle.load(f) / 255
# X_val = np.array(X_v)
# with open('../The_Pose/tfrecord/y_new_val.pickle', 'rb') as f:
#     y_v = pickle.load(f)
# y_val = np.array(y_v)

# y_tr = y_val[0*32:(0+1)*32]
#
# output = tc.layers.flatten(y_tr)
#
# keypoint_xy_class_probability = output[:, :2400]
# box_wh = output[:, 2400:]
#
# mp_out = tf.concat([keypoint_xy_class_probability, box_wh], 1)
#
# output = tf.reshape(mp_out, [-1,10,10,26])
#
# init_op = tf.global_variables_initializer()
# init_l = tf.local_variables_initializer()
#
# with tf.Session() as sess:
#     sess.run(init_op)
#     sess.run(init_l)
#     y_true = sess.run(output)
#
# print np.array_equal(y_true, y_tr)
#
# print np.allclose(y_true, y_tr)
# print np.allclose(np.reshape(y_true, [-1,26]), np.reshape(y_tr, [-1,26]))
