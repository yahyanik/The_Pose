from __future__ import division

import numpy as np
import cv2
from pycocotools.coco import COCO
from model_normal import *
from Metrics1 import read_and_decode


def DataRead (addr, type):
    dataDir = '..'
    dataType = type
    coco_kps = COCO(addr.format(dataDir, dataType))
    imgIds = coco_kps.getImgIds(catIds=coco_kps.getCatIds(catNms=['person']))  # get image IDs that have human in them
    return coco_kps, imgIds

def vis_parallel():

    # x = tf.placeholder(dtype=tf.float32, shape=[None, 320, 320, 3])
    # model = MobileNetV2_normal(num_to_reduce=2, is_training=True, input_size=320, input_placeholder=x)

    coco_kps, imgIds = DataRead('../The_Pose/database/coco/annotations/person_keypoints_train2017.json', 'train2017')

    for i in range (43, 44):
        img = coco_kps.loadImgs(imgIds[i])[0]
    # print img

        imgFile = cv2.imread('../The_Pose/database/coco/images/train2017/' + img['file_name'])
        annIds = coco_kps.getAnnIds(imgIds=imgIds[i], iscrowd=None)

    # taking the labels out of the dataset and showing the image
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

            # feed_image = cv2.imread('../The_Pose/database/coco/images/train2017/' + img['file_name'])
            # feed_image = cv2.resize(feed_image, (320, 320))
            # tfrecords_filename = '../The_Pose/tfrecord/keypoint_train_new.tfrecords'
            # filename_queue = tf.train.string_input_producer([tfrecords_filename])
            # image, annotation = read_and_decode(320, filename_queue, 64, 10)
            # saver = tf.train.Saver()
            # with tf.Session() as sess:
            #     sess.run(tf.global_variables_initializer())
            #     print 'before img'
            #     img, anno = sess.run([image, annotation])
            #     print 'img back'

        # saver.restore(sess, "./model_1280relu_7/tmp/model.ckpt-241")
        #
        # feed_image = feed_image.reshape([1, 320, 320, 3])
        # y_prediction = sess.run(model.output, feed_dict={x: img})  # model results as y_hat
        #
        # print y_prediction.shape
        # print np.unique(y_prediction)
        #
        # print np.unicode(anno)

        # img = img *255
        # print img


    # print image sized info

    # y_pre_reshaped = np.reshape(y_prediction, [-1, 26])
    #
    # head = np.greater (y_pre_reshaped[:,2], 0.2)
    # l_arm = np.greater (y_pre_reshaped[:,5], 0.5)
    # r_arm = np.greater (y_pre_reshaped[:,8], 0.5)
    # l_hand = np.greater (y_pre_reshaped[:,11], 0.5)
    # r_hand = np.greater (y_pre_reshaped[:,14], 0.5)
    # l_finger = np.greater (y_pre_reshaped[:,17], 0.5)
    # r_finger = np.greater (y_pre_reshaped[:,20], 0.5)
    # human = np.greater (y_pre_reshaped[:,21], 0.5)
    #
    # print y_pre_reshaped[:, 11]
    # print head

        # for ii in range(10):
        #     for jj in range(10):
        #         if y_prediction[:,ii,jj,2] >= 0.15:
        #             print 'head detected'
        #         if y_prediction[:,ii,jj,5] >= 0.15:
        #             print 'l_sholder detected'
        #         if y_prediction[:,ii,jj,8] >= 0.15:
        #             print 'r_sholder detected'
        #         if y_prediction[:,ii,jj,11] >= 0.15:
        #             print 'l_arm detected'
        #         if y_prediction[:,ii,jj,14] >= 0.15:
        #             print 'r_arm detected'
        #         if y_prediction[:,ii,jj,17] >= 0.15:
        #             print 'l_hand detected'
        #         if y_prediction[:,ii,jj,20] >= 0.15:
        #             print 'r_hand detected'

#####################################################################################################################

def train (imagesize = 320, batch_size = 1, num_threads = 10):

    tfrecords_filename = '../The_Pose/tfrecord/keypoint_train_new.tfrecords'
    filename_queue = tf.train.string_input_producer([tfrecords_filename])
    image, annotation = read_and_decode(imagesize, filename_queue, batch_size, num_threads)
    x = tf.placeholder(dtype=tf.float32, shape=[None, imagesize, imagesize, 3],name='x')
    model = MobileNetV2_normal(num_to_reduce=32, is_training=True, input_size=320, input_placeholder=x)
    saver = tf.train.Saver()
    init_op = tf.global_variables_initializer()
    init_l = tf.local_variables_initializer()

    with tf.Session() as sess:
        sess.run(init_op)
        sess.run(init_l)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        saver.restore(sess, "./model_normal_2.2.32_1/tmp/model.ckpt-361")
        for i in range(43,50):
            img, anno = sess.run([image, annotation])
            y_prediction = sess.run(model.output, feed_dict={x: img})  # model results as y_hat
        coord.request_stop()
        coord.join(threads)

    print y_prediction[0, :, :, 21]
    print np.max(y_prediction[0, :, :, 21])
    print anno[0, :, :, 21]
    for i in range(10):
        for j in range(10):
            if np.greater(y_prediction[0,i,j,2], 0.5):      #if there is a head
                xh = int((32*i)+y_prediction[0,i,j,0])
                yh = int((32*j)+y_prediction[0,i,j,1])
                cv2.circle(img[0, :, :, :], (xh, yh), 10, (0, 0, 255), -1)

            if np.greater(y_prediction[0,i,j,5], 0.5):      #if there is a left sholder
                xh = int((32*i)+y_prediction[0,i,j,3])
                yh = int((32*j)+y_prediction[0,i,j,4])
                cv2.circle(img[0, :, :, :], (xh, yh), 10, (0, 0, 255), -1)

            if np.greater(y_prediction[0,i,j,8], 0.5):      #if there is a right sholder
                xh = int((32*i)+y_prediction[0,i,j,6])
                yh = int((32*j)+y_prediction[0,i,j,7])
                cv2.circle(img[0, :, :, :], (xh, yh), 10, (0, 0, 255), -1)

            if np.greater(y_prediction[0,i,j,11], 0.5):      #if there is a left elbow
                xh = int((32*i)+y_prediction[0,i,j,9])
                yh = int((32*j)+y_prediction[0,i,j,10])
                cv2.circle(img[0, :, :, :], (xh, yh), 10, (0, 0, 255), -1)

            if np.greater(y_prediction[0,i,j,14], 0.5):      #if there is a right elbow
                xh = int((32*i)+y_prediction[0,i,j,12])
                yh = int((32*j)+y_prediction[0,i,j,13])
                cv2.circle(img[0, :, :, :], (xh, yh), 10, (0, 0, 255), -1)

            if np.greater(y_prediction[0,i,j,17], 0.5):      #if there is a left hand
                xh = int((32*i)+y_prediction[0,i,j,15])
                yh = int((32*j)+y_prediction[0,i,j,16])
                cv2.circle(img[0, :, :, :], (xh, yh), 10, (0, 0, 255), -1)

            if np.greater(y_prediction[0,i,j,20], 0.5):      #if there is a right hand
                xh = int((32*i)+y_prediction[0,i,j,18])
                yh = int((32*j)+y_prediction[0,i,j,19])
                cv2.circle(img[0, :, :, :], (xh, yh), 10, (0, 0, 255), -1)

            if np.greater(y_prediction[0,i,j,21], 0.5):      #if there is a person
                xh = (32*i)+y_prediction[0,i,j,22]
                yh = (32*j)+y_prediction[0,i,j,23]
                print 32 * y_prediction[0, i, j, 24]
                W = int(32*y_prediction[0,i,j,24])
                if W >10:
                    W = 10
                H = int(32*y_prediction[0,i,j,25])
                if H>10:
                    H = 10
                xh1 = int(xh-W/2)
                xh2 = int(xh + W / 2)
                yh1 = int(yh - H / 2)
                yh2 = int(yh + H / 2)
                cv2.rectangle(img[0, :, :, :], (xh1,yh1), (xh2,yh2), (0, 255, 0),3)
    cv2.imshow('track', img[0, :, :, :])
    cv2.waitKey(0)



if __name__ == '__main__':

    train()