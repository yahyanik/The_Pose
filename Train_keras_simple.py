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

# Seyed Yahya Nikouei Summer 2019, parallel learning the MNV2 on COCO dataset for human detection and gesture extractoin
# This one goes with all the _8 files to train a network with just the detection of the object in a window

# def train (epoch = 5, loop=50, learning_rate = 0.0001, drop_out = 0.7, regular_fac = 0.01, num_to_reduce=32, imagesize = 320, batch_size = 32, num_threads = 10, num_gpus = 2):
#
#     FolderName = './normal{}_69.{}_1280_{}_0.5_5'.format(batch_size, learning_rate, regular_fac)
#     # checkpoint_path = FolderName + "/tmp/model_epoch{epoch:02d}_loop00.ckpt"
#     tensorboard_name = 'TB_keypoint_keras_{}'.format(int(time.time()))
#     # file_checkpoint = FolderName + "/keras/model_epoch{epoch:02d}.h5"
#     try:
#         # os.makedirs(FolderName+'/keras')
#         os.makedirs(FolderName + '/tmp')
#     except:
#         shutil.rmtree(FolderName)
#         # os.makedirs(FolderName + '/keras')
#         os.makedirs(FolderName + '/tmp')
#
#     tfrecords_filename = '../The_Pose/tfrecord/DATASET_COCO.tfrecords'
#     tfrecords_filename_val = '../The_Pose/tfrecord/DATASET_VAL.tfrecords'
#
#     # checkpoint_dir = os.path.dirname(checkpoint_path)
#
#     # cp_callback = k.callbacks.ModelCheckpoint(checkpoint_path, verbose=1,period=5, save_weights_only=False)
#     # checkpoint = ModelCheckpoint(file_checkpoint, verbose=1, load_weights_on_restart=False,period=5)
#
#     num_batch = int(64115 / batch_size)
#     num_batch_val = int(2693/batch_size)
#
#     image, annotation = read_image_tf_data(imagesize, tfrecords_filename, batch_size, num_threads)
#     image_val, annotation_val = read_image_tf_data(imagesize, tfrecords_filename_val, batch_size, num_threads)
#
#     model_obj = MobileNetV2_normal_keras(num_to_reduce=num_to_reduce, drop_fac = drop_out, head_is_training=False, regular_fac=regular_fac,
#                                          layers_to_fine_tune=156, include_top=False, train_layers=False)
#
#     metric = metric_custom()
#     # irre_metric = Out_Metric()
#     metric_list = [metric.RECAL, metric.PERCISION, metric.Distance_parallel, metric.get_max, metric.get_min]
#
#     parallel_model = multi_gpu_model(model_obj.model, gpus=num_gpus)
#     parallel_model.compile(optimizer=tf.keras.optimizers.Adam(lr=learning_rate),
#                   loss=my_cost_MSE, metrics=metric_list)
#
#     # print (parallel_model.summary())
#     # print("Number of layers in the model: ", len(parallel_model.layers), 'and 155 in the base model')
#
#     tensorboard = TensorBoard(log_dir=FolderName+'/logs/'+tensorboard_name)
#     # callbacks_list = [tensorboard, cp_callback]
#     # checkpoint.on_epoch_end(epoch, logs=[metric.RECAL_8, metric.PERCISION_8])
#
#     # history = parallel_model.fit(image, annotation, epochs=epoch, batch_size=batch_size, steps_per_epoch=num_batch,
#     #                              validation_data=(image_val, annotation_val), validation_steps=num_batch_val,
#     #                              callbacks=callbacks_list)
#     init_op = tf.global_variables_initializer()
#     init_l = tf.local_variables_initializer()
#     tf.logging.set_verbosity(tf.logging.ERROR)
#
#     with tf.Session() as sess:
#         sess.run(init_op)
#         sess.run(init_l)
#         for i in range(loop):
#             checkpoint_path = FolderName + "/tmp/model_epoch{epoch:02d}_loop"+str(i)+".ckpt"
#             checkpoint_dir = os.path.dirname(checkpoint_path)
#             cp_callback = k.callbacks.ModelCheckpoint(checkpoint_path, verbose=1, period=5, save_weights_only=False)
#             callbacks_list = [tensorboard, cp_callback]
#             history = parallel_model.fit(image, annotation, epochs=epoch, batch_size=batch_size, steps_per_epoch=num_batch,
#                                      validation_data=(image_val,annotation_val), validation_steps=num_batch_val,callbacks=callbacks_list)
#
#
#             sk_map = 0
#             for bat_v in range(int(2693 / batch_size)):
#                 img_val, anno_val = sess.run([image_val, annotation_val])
#                 pre_val = parallel_model.predict(img_val)
#
#                 y_true_val = np.reshape(anno_val, [-1, 26])
#                 y_hat_val = np.reshape(pre_val, [-1, 26])
#                 sk_map1 = average_precision_score(np.sign(y_true_val[:, 2]).astype(int), y_hat_val[:, 2])
#                 sk_map2 = average_precision_score(np.sign(y_true_val[:, 5]).astype(int), y_hat_val[:, 5])
#                 sk_map3 = average_precision_score(np.sign(y_true_val[:, 8]).astype(int), y_hat_val[:, 8])
#                 sk_map4 = average_precision_score(np.sign(y_true_val[:, 11]).astype(int), y_hat_val[:, 11])
#                 sk_map5 = average_precision_score(np.sign(y_true_val[:, 14]).astype(int), y_hat_val[:, 14])
#                 sk_map6 = average_precision_score(np.sign(y_true_val[:, 17]).astype(int), y_hat_val[:, 17])
#                 sk_map7 = average_precision_score(np.sign(y_true_val[:, 20]).astype(int), y_hat_val[:, 20])
#                 sk_map8 = average_precision_score(np.sign(y_true_val[:, 21]).astype(int), y_hat_val[:, 21])
#
#                 sk_map += (sk_map1 + sk_map2 + sk_map3 + sk_map4 + sk_map5 + sk_map6 + sk_map7 + sk_map8) / 8.0
#
#             print('epoch =', (i+1)*5, 'learning_rate =', learning_rate, 'regular_fac =', regular_fac)
#             print('sk-mAP test', sk_map / int(2693 / batch_size))
#
#
#
#
#     # print(irre_metric.get_data())
#     #
#     # # y_pred = parallel_model.predict(image_val, batch_size=batch_size)
#     #
#     # with open(FolderName+'/tmp/logs.txt', 'wb') as file_pi:
#     #     pickle.dump(history.history, file_pi)


def train1 (epoch = 120, layers_not_training =117, learning_rate = 0.0005, drop_out = 0.6, regular_fac = 0.1, num_to_reduce=32, imagesize = 320, batch_size = 64, num_threads = 10, num_gpus = 2):
    #155 is total, 144 is before 320 and 117 is before 63

    FolderName = './normal{}_113.{}_{}_{}_0.5_5_GPU'.format(batch_size, learning_rate, layers_not_training,regular_fac)
    tensorboard_name = 'TB_keypoint_keras_{}'.format(int(time.time()))
    try:
        os.makedirs(FolderName + '/tmp')
    except:
        shutil.rmtree(FolderName)
        os.makedirs(FolderName + '/tmp')

    tfrecords_filename = '../The_Pose/tfrecord/DATASET_BLUR_COCO.tfrecords'
    tfrecords_filename_val = '../The_Pose/tfrecord/DATASET_BLUR_VAL.tfrecords'

    num_batch = int(64115 / batch_size)
    num_batch_val = int(2693/batch_size)

    image, annotation= read_image_tf_data(imagesize, tfrecords_filename, batch_size, num_threads)
    image_val, annotation_val = read_image_tf_data(imagesize, tfrecords_filename_val, batch_size, num_threads)

    model_obj = MobileNetV2_normal_keras(num_to_reduce=num_to_reduce, drop_fac=drop_out, head_is_training=False,
                                         regular_fac=regular_fac,
                                         layers_to_fine_tune=layers_not_training, include_top=False, train_layers=False)

    print (model_obj.model.summary())
    print('len(model.trainable_variables)', len(model_obj.model.layers))

    metric = metric_custom()
    metric_list = [metric.RECAL, metric.PERCISION, metric.Distance_parallel, metric.get_max, metric.get_min,
                   metric.recall_body, metric.percision_body, metric.recall_detection, metric.percision_detection]

    model_obj.model.compile(optimizer=tf.keras.optimizers.Adam(lr=learning_rate),
                  loss=my_cost_MSE, metrics=metric_list)


    tensorboard = TensorBoard(log_dir=FolderName+'/logs/'+tensorboard_name)

    checkpoint_path = FolderName + "/tmp/model_epoch{epoch:02d}.hdf5"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    cp_callback = k.callbacks.ModelCheckpoint(checkpoint_path, verbose=1, period=5, save_weights_only=False)
    callbacks_list = [tensorboard, cp_callback]
    print(model_obj.model.summary())
    print('len(model.trainable_variables)', len(model_obj.model.layers))
    history = model_obj.model.fit(image, annotation, epochs=epoch, batch_size=batch_size, steps_per_epoch=num_batch,
                             validation_data=(image_val,annotation_val), validation_steps=num_batch_val,callbacks=callbacks_list)
    init_op = tf.global_variables_initializer()
    init_l = tf.local_variables_initializer()
    tf.logging.set_verbosity(tf.logging.ERROR)
    with tf.Session() as sess:
        sess.run(init_op)
        sess.run(init_l)

        sk_map = 0
        # for bat_v in range(int(2693/batch_size)):
        for bat_v in range(1):
            img_val, anno_val = sess.run([image, annotation])
            # print (np.max(img_val[1,:,:,:]), np.min(img_val[1,:,:,:]))
            print(img_val.shape)
            # mean1 = [np.sum(img_val[:,:,:,0])/(shape[0]*shape[1]*shape[2]),
            #         np.sum(img_val[:,:,:,1])/(shape[0]*shape[1]*shape[2]),
            #         np.sum(img_val[:,:,:,2])/(shape[0]*shape[1]*shape[2])]
            # print (mean, std)
            # cv2.imshow('img', img_val[1,:,:,:])
            # cv2.waitKey(0)
            # print (images_val.shape)
            # print(label_val.shape)
            # print(mean_val.shape)
            # print(std_val.shape)
            # print (mean_val, std_val)


            pre_val = model_obj.model.predict(img_val)
            print(pre_val.shape)

            y_true_val = np.reshape(anno_val, [-1, 26])
            y_hat_val = np.reshape(pre_val, [-1, 26])
            sk_map1 = average_precision_score(np.sign(y_true_val[:, 2]).astype(int), y_hat_val[:, 2])
            sk_map2 = average_precision_score(np.sign(y_true_val[:, 5]).astype(int), y_hat_val[:, 5])
            sk_map3 = average_precision_score(np.sign(y_true_val[:, 8]).astype(int), y_hat_val[:, 8])
            sk_map4 = average_precision_score(np.sign(y_true_val[:, 11]).astype(int), y_hat_val[:, 11])
            sk_map5 = average_precision_score(np.sign(y_true_val[:, 14]).astype(int), y_hat_val[:, 14])
            sk_map6 = average_precision_score(np.sign(y_true_val[:, 17]).astype(int), y_hat_val[:, 17])
            sk_map7 = average_precision_score(np.sign(y_true_val[:, 20]).astype(int), y_hat_val[:, 20])
            sk_map8 = average_precision_score(np.sign(y_true_val[:, 21]).astype(int), y_hat_val[:, 21])

            sk_map += (sk_map1 + sk_map2 + sk_map3 + sk_map4 + sk_map5 + sk_map6 + sk_map7 + sk_map8) / 8.0

        print('learning_rate =', learning_rate, 'regular_fac =', regular_fac)
        print('sk-mAP test', sk_map / int(2693 / batch_size))



if __name__ == '__main__':
    train1(imagesize=320, num_threads=10) # in the parallel we are still doing batch of 32