import numpy as np
import cv2
import tensorflow as tf
import sklearn.metrics as metrics
from pycocotools.coco import COCO
from model_normal import *
import matplotlib.pyplot as plt




def DataRead (addr, type):
    dataDir = '..'
    dataType = type
    coco_kps = COCO(addr.format(dataDir, dataType))
    imgIds = coco_kps.getImgIds(catIds=coco_kps.getCatIds(catNms=['person']))  # get image IDs that have human in them
    return (coco_kps, imgIds)

x = tf.placeholder(dtype=tf.float32, shape=[None, 320, 320, 3],name='x')
model = MobileNetV2_1280relu_new(2, False, 320, x)

coco_kps, imgIds = DataRead('../The_Pose/database/coco/annotations/person_keypoints_train2017.json', 'val2017')

for i in range (32,35):
    img = coco_kps.loadImgs(imgIds[i])[0]
    # print img


    imgFile = cv2.imread('../The_Pose/database/coco/images/train2017/' + img['file_name'])
    annIds = coco_kps.getAnnIds(imgIds=imgIds[i], iscrowd=None)
    for id in annIds:
        anns = coco_kps.loadAnns(id)
        key = anns[0]['keypoints']
        bbox = anns[0]['bbox']
        center = [bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2]

        cv2.circle(imgFile, (key[0], key[1]), 2, (0, 0, 255), -1)
        cv2.circle(imgFile, (key[15],key[16]), 2, (0, 0, 255), -1)
        cv2.circle(imgFile, (key[18],key[19]), 2, (0, 0, 255), -1)
        cv2.circle(imgFile, (key[21],key[22]), 2, (0, 0, 255), -1)
        cv2.circle(imgFile, (key[24],key[25]), 2, (0, 0, 255), -1)
        cv2.circle(imgFile, (key[27], key[28]), 2, (0, 0, 255), -1)
        cv2.circle(imgFile, (key[30], key[31]), 2, (0, 0, 255), -1)
        cv2.circle(imgFile, (int(center[0]), int(center[1])),4,(255,0,200),-1)
        cv2.rectangle(imgFile, (int(bbox[0]), int(bbox[1])), (int(bbox[0]+bbox[2]), int(bbox[1]+bbox[3])), (0,255,0),1)






    # print imgFile.shape
    feed_image = cv2.resize(imgFile, (320,320))
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        saver.restore(sess, "./model_1280relu_7/tmp/model.ckpt-71")

        feed_image = feed_image.reshape([1,320,320,3])
        y_prediction = sess.run(model.output, feed_dict={model.input: feed_image})  # model results as y_hat

        print y_prediction.shape
        print np.unique(y_prediction)



    # fpr, tpr, threshold = metrics.roc_curve(y_test, y_prediction)
    # roc_auc = metrics.auc(fpr, tpr)
    # plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)



    cv2.imshow('image1', imgFile)
    cv2.waitKey(0)


