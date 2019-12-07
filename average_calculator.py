from __future__ import division
import tensorflow as tf
import tensorflow.keras as k
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import ModelCheckpoint
from model_normal_keras import *
import time
import shutil
import cv2
import os
from cost_keras import *
from Make_parallel import *
from Metrics_keras import *
from tensorflow.keras.utils import multi_gpu_model


def train1 (epoch = 120, layers_not_training =117, learning_rate = 0.0005, drop_out = 0.6, regular_fac = 0.1, num_to_reduce=32, imagesize = 320, batch_size = 64, num_threads = 10, num_gpus = 2):

    tfrecords_filename_val = '../The_Pose/tfrecord/DATASET_BLUR_VAL.tfrecords'

    num_batch_val = int(2693/batch_size)

    image_val, annotation_val = read_image_tf_data(imagesize, tfrecords_filename_val, batch_size, num_threads)

    model_obj = MobileNetV2_normal_keras(num_to_reduce=num_to_reduce, drop_fac=drop_out, head_is_training=False,
                                         regular_fac=regular_fac,
                                         layers_to_fine_tune=layers_not_training, include_top=False, train_layers=False)
    metric = metric_custom()
    metric_list = [metric.RECAL, metric.PERCISION, metric.Distance_parallel, metric.get_max, metric.get_min,
                   metric.recall_body, metric.percision_body, metric.recall_detection, metric.percision_detection,
                   metric.avg_RECAL, metric.avg_PERCISION, metric.avg_recall_body, metric.avg_recall_detection,
                   metric.avg_percision_body,metric.avg_percision_detection]

    parallel_model = multi_gpu_model(model_obj.model, gpus=num_gpus)
    parallel_model.compile(optimizer=tf.keras.optimizers.Adam(lr=learning_rate),
                  loss=my_cost_MSE, metrics=metric_list)

    file_checkpoint = './normal64_111.0.0005_117_0.1_0.5_5_1024_final/tmp/model_epoch95.hdf5'
    parallel_model.load_weights(file_checkpoint)

    # cp_callback = k.callbacks.ModelCheckpoint(checkpoint_path, verbose=1, period=5, save_weights_only=False)
    # callbacks_list = [cp_callback]
    history = parallel_model.predict(image_val, annotation_val, batch_size=batch_size, steps=num_batch_val)



    # init_op = tf.global_variables_initializer()
    # init_l = tf.local_variables_initializer()
    # tf.logging.set_verbosity(tf.logging.ERROR)
    # with tf.Session() as sess:
    #     sess.run(init_op)
    #     sess.run(init_l)
    #
    #     sk_map = 0
    #     # for bat_v in range(int(2693/batch_size)):
    #     for bat_v in range(1):
    #         img_val, anno_val = sess.run([image_val, annotation_val])
    #         # print (np.max(img_val[1,:,:,:]), np.min(img_val[1,:,:,:]))
    #         print(img_val.shape)
    #
    #         pre_val = parallel_model.predict(img_val)
    #         print(pre_val.shape)
    #
    #         y_true_val = np.reshape(anno_val, [-1, 26])
    #         y_hat_val = np.reshape(pre_val, [-1, 26])
    #         sk_map1 = average_precision_score(np.sign(y_true_val[:, 2]).astype(int), y_hat_val[:, 2])
    #         sk_map2 = average_precision_score(np.sign(y_true_val[:, 5]).astype(int), y_hat_val[:, 5])
    #         sk_map3 = average_precision_score(np.sign(y_true_val[:, 8]).astype(int), y_hat_val[:, 8])
    #         sk_map4 = average_precision_score(np.sign(y_true_val[:, 11]).astype(int), y_hat_val[:, 11])
    #         sk_map5 = average_precision_score(np.sign(y_true_val[:, 14]).astype(int), y_hat_val[:, 14])
    #         sk_map6 = average_precision_score(np.sign(y_true_val[:, 17]).astype(int), y_hat_val[:, 17])
    #         sk_map7 = average_precision_score(np.sign(y_true_val[:, 20]).astype(int), y_hat_val[:, 20])
    #         sk_map8 = average_precision_score(np.sign(y_true_val[:, 21]).astype(int), y_hat_val[:, 21])
    #
    #         sk_map += (sk_map1 + sk_map2 + sk_map3 + sk_map4 + sk_map5 + sk_map6 + sk_map7 + sk_map8) / 8.0
    #
    #     print('learning_rate =', learning_rate, 'regular_fac =', regular_fac)
    #     print('sk-mAP test', sk_map / int(2693 / batch_size))



if __name__ == '__main__':
    train1(imagesize=320, num_threads=10) # in the parallel we are still doing batch of 32