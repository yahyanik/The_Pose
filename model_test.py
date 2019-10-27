from __future__ import division
import tensorflow as tf
import tensorflow.keras as k
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import ModelCheckpoint
from model_normal_keras import *
from openpose import *
import time
import datetime
from VGG import *
import shutil
import os
from cost_keras import *
from Make_parallel import *
from Metrics_keras import *
from tensorflow.keras.utils import multi_gpu_model
import cv2


def test_mAP (epoch = 120, layers_not_training =117, learning_rate = 0.0001, drop_out = 1, regular_fac = 0.1, num_to_reduce=32, imagesize = 320, batch_size = 32, num_threads = 10, num_gpus = 2):


    FolderName = './normal{}_88.{}_{}_{}_0.5_5'.format(batch_size, learning_rate, layers_not_training,regular_fac)
    file_checkpoint = FolderName + "/tmp/model_epoch120.ckpt"
    tensorboard_name = 'TB_VALIDATE_keypoint_keras_{}'.format(int(time.time()))

    tfrecords_filename_val = '../The_Pose/tfrecord/DATASET_BLUR_VAL.tfrecords'
    image_val, annotation_val = read_image_tf_data(320, tfrecords_filename_val, batch_size, num_threads)
    metric = metric_custom()
    metric_list = [metric.RECAL, metric.PERCISION, metric.Distance_parallel, metric.get_max, metric.get_min,
                   metric.recall_body, metric.percision_body, metric.recall_detection, metric.percision_detection]

    model_obj = MobileNetV2_normal_keras(num_to_reduce=num_to_reduce, drop_fac=drop_out, head_is_training=False,
                                         regular_fac=regular_fac,
                                         layers_to_fine_tune=layers_not_training, include_top=False, train_layers=False)
    openpose=get_testing_model()
    vg = VGG_16()


    parallel_model = multi_gpu_model(model_obj.model, gpus=num_gpus)
    # model_obj.model.compile(optimizer=tf.keras.optimizers.Adam(lr=learning_rate),
    #                        loss=my_cost_MSE, metrics=metric_list)
    # model_obj.model.load_weights(file_checkpoint)
    tensorboard = TensorBoard(log_dir=FolderName + '/logs/' + tensorboard_name)
    num_batch_val = int(2693 / batch_size)
    init_op = tf.global_variables_initializer()
    init_l = tf.local_variables_initializer()
    tf.logging.set_verbosity(tf.logging.ERROR)
    with tf.Session() as sess:
        sess.run(init_op)
        sess.run(init_l)

        sk_map = 0
        for bat_v in range(100):
            img_val, anno_val = sess.run([image_val, annotation_val])
            tik = datetime.datetime.now()
            res = model_obj.model.predict(img_val, batch_size=1, verbose=1, steps=100)
            res = parallel_model.predict(img_val, batch_size=1, verbose=1, steps=100)
            res = vg.predict(img_val, batch_size=1, verbose=1, steps=100)
            res = openpose.predict(img_val, batch_size=1, verbose=1, steps=100)

            tok=datetime.datetime.now()
            print(tok-tik)
    # print(res)
    # loss, acc = parallel_model.evaluate(image_val, annotation_val, batch_size=batch_size, steps=num_batch_val, verbose=1, callbacks=[tensorboard])

    print("Restored model, accuracy: {:5.2f}%".format(100 * acc))





if __name__ == '__main__':
    test_mAP()