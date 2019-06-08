from __future__ import division
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
from model_normal_keras import *
import pickle
# import os
import datetime
from Metrics1 import *
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'


def train (epoch = 500, learning_rate = 0.005, regular_fac = 0.002, num_to_reduce=2, imagesize = 320, batch_size = 16, num_threads = 10, num_gpus = 2):

    FolderName = './model_normal{}_2.29.320_52_{}'.format(batch_size, regular_fac)
    num_batch = int(81366 / batch_size)     #64115 is the number of examples in the dataset
    # num_batch = int(64115 / batch_size)
    # tfrecords_filename = '../The_Pose/tfrecord/DATASET_COCO.tfrecords'
    tfrecords_filename = '../The_Pose/tfrecord/DATASET_gn.tfrecords'
    tfrecords_filename_val = '../The_Pose/tfrecord/keypoint_val_Corrected.1.tfrecords'
    filename_queue = tf.train.string_input_producer([tfrecords_filename])
    filename_queue_val = tf.train.string_input_producer([tfrecords_filename_val])
    image, annotation = read_and_decode(imagesize, filename_queue, batch_size, num_threads)
    image_val, annotation_val = read_and_decode(imagesize, filename_queue_val, batch_size, num_threads)


    init_op = tf.global_variables_initializer()
    init_l = tf.local_variables_initializer()
    tf.logging.set_verbosity(tf.logging.ERROR)

    with tf.Session() as sess:

        sess.run(init_op)
        sess.run(init_l)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        koli = 0
        onesi = 0
        zerosi = 0
        ones_in_class = 0
        koli_in_class = 0
        zeros_in_class = 0

        for bat in range(num_batch):
        # for bat in range(1):
            img, anno = sess.run([image, annotation])
            onesi += np.count_nonzero(anno)
            koli += (anno.shape[0]*anno.shape[1]*anno.shape[2]*anno.shape[3])
            zerosi += (anno.shape[0]*anno.shape[1]*anno.shape[2]*anno.shape[3]) - np.count_nonzero(anno)
            anno1 = np.reshape(anno, [1600, 26])
            k = np.count_nonzero(anno1[:,2])+np.count_nonzero(anno1[:,5])+np.count_nonzero(anno1[:,8])+np.count_nonzero(anno1[:,11])+np.count_nonzero(anno1[:,14])+ \
            np.count_nonzero(anno1[:, 17]) +np.count_nonzero(anno1[:,20])+np.count_nonzero(anno1[:,21])
            w = 1600*8
            ones_in_class += k
            koli_in_class += w
            zeros_in_class += w - k




        print('ones:', onesi, 'zeros', zerosi, 'kol', koli)
        check = True if onesi+zerosi == koli else False
        print (check)
        print (num_batch, batch_size)
        print ('ones_in_class',ones_in_class, 'koli_in_class', koli_in_class, 'zeros_in_class', zeros_in_class)

        coord.request_stop()
        coord.join(threads)

if __name__ == '__main__':
    train(imagesize=320, num_threads=10) # in the parallel we are still doing batch of 32