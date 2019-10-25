from __future__ import division
import tensorflow as tf
import tensorflow.keras as k
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import ModelCheckpoint
from model_normal_keras import *
import time
import shutil
import os
from cost_keras import *
from Make_parallel import *
from Metrics_keras import *
from tensorflow.keras.utils import multi_gpu_model
import cv2


def test_mAP (learning_rate = 0.0001, layers_not_training = 117,regular_fac = 0.001, num_to_reduce=32, imagesize = 320, batch_size = 32, num_threads = 10, num_gpus = 2):


    FolderName = './normal{}_88.{}_{}_{}_0.5_5'.format(batch_size, learning_rate, layers_not_training,regular_fac)
    file_checkpoint = FolderName + "/tmp/model_epoch120.ckpt"


    tfrecords_filename_val = '../The_Pose/tfrecord/DATASET_BLUR_VAL.tfrecords'
    image_val, annotation_val = read_image_tf_data(imagesize, tfrecords_filename_val, batch_size, num_threads)
    metric = metric_custom()

    model_obj = MobileNetV2_normal_keras(num_to_reduce=num_to_reduce, head_is_training=False, regular_fac=regular_fac,
                                         layers_to_fine_tune=150, include_top=False, fireezed_layers=False)

    parallel_model = multi_gpu_model(model_obj.model, gpus=num_gpus)
    parallel_model.compile(optimizer=tf.keras.optimizers.Adam(lr=learning_rate),
                           loss=my_cost_MSE, metrics=[metric.RECAL, metric.PERCISION, metric.Distance_parallel])
    parallel_model.load_weights(file_checkpoint)


    sk_mAP_result = metric.sk_mAP(parallel_model, batch_size, image_val, annotation_val, 2693)



if __name__ == '__main__':
    test_mAP()