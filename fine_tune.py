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



def train1 (epoch = 40, layers_not_training =14, learning_rate = 0.0000001, drop_out = 0.6, regular_fac = 0.1, num_to_reduce=32, imagesize = 320, batch_size = 64, num_threads = 10, num_gpus = 2):
    #155 is total, 144 is before 320 and 117 is before 63 or 14

    FolderName = './normal{}_fine_113_1.{}_{}_{}_0.5_5_average'.format(batch_size, learning_rate, layers_not_training,regular_fac)
    tensorboard_name = 'TB_keypoint_keras_{}'.format(int(time.time()))
    try:
        os.makedirs(FolderName + '/tmp')
    except:
        shutil.rmtree(FolderName)
        os.makedirs(FolderName + '/tmp')

    tfrecords_filename = '../The_Pose/tfrecord/DATASET_BLUR_COCO.tfrecords'
    tfrecords_filename_val = '../The_Pose/tfrecord/DATASET_BLUR_VAL.tfrecords'

    file_checkpoint = './normal64_112.0.0005_117_0.1_0.5_5_GPU/tmp/model_epoch120.hdf5'

    num_batch = int(64115 / batch_size)
    num_batch_val = int(2693/batch_size)

    image, annotation= read_image_tf_data(imagesize, tfrecords_filename, batch_size, num_threads)
    image_val, annotation_val = read_image_tf_data(imagesize, tfrecords_filename_val, batch_size, num_threads)

    model_obj = MobileNetV2_normal_keras(num_to_reduce=num_to_reduce, drop_fac=drop_out, head_is_training=False,
                                         regular_fac=regular_fac,
                                         layers_to_fine_tune=layers_not_training, include_top=False, train_layers=False)

    metric = metric_custom()
    metric_list = [metric.RECAL, metric.PERCISION, metric.Distance_parallel, metric.get_max, metric.get_min,
                   metric.recall_body, metric.percision_body, metric.recall_detection, metric.percision_detection]

    # parallel_model = multi_gpu_model(model_obj.model, gpus=num_gpus)
    print (model_obj.model.summary())
    print('len(model.trainable_variables)', len(model_obj.model.layers))
    model_obj.model.load_weights(file_checkpoint)
    model_obj.model.compile(optimizer=tf.keras.optimizers.Adam(lr=learning_rate),
                  loss=my_cost_MSE, metrics=metric_list)


    tensorboard = TensorBoard(log_dir=FolderName+'/logs/'+tensorboard_name)

    checkpoint_path = FolderName + "/tmp/model_epoch{epoch:02d}.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    cp_callback = k.callbacks.ModelCheckpoint(checkpoint_path, verbose=1, period=5, save_weights_only=False)
    callbacks_list = [tensorboard, cp_callback]
    print(model_obj.model.summary())
    print('len(model.trainable_variables)', len(model_obj.model.layers))
    history = model_obj.model.fit(image, annotation, epochs=epoch, batch_size=batch_size, steps_per_epoch=num_batch,
                             validation_data=(image_val,annotation_val), validation_steps=num_batch_val,callbacks=callbacks_list)



if __name__ == '__main__':
    train1(imagesize=320, num_threads=10) # in the parallel we are still doing batch of 32