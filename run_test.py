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


def run_test ():

    FolderName = './normal{}_88.{}_{}_{}_0.5_5'.format(batch_size, learning_rate, layers_not_training,regular_fac)
    tensorboard_name = 'TB_keypoint_keras_{}'.format(int(time.time()))
    try:
        os.makedirs(FolderName + '/tmp')
    except:
        shutil.rmtree(FolderName)
        os.makedirs(FolderName + '/tmp')

    tfrecords_filename = '../The_Pose/tfrecord/DATASET_BLUR_COCO.tfrecords'
    tfrecords_filename_val = '../The_Pose/tfrecord/DATASET_BLUR_VAL.tfrecords'



    # image, annotation= read_image_tf_data(imagesize, tfrecords_filename, batch_size, num_threads)
    image_val, annotation_val = read_image_tf_data(imagesize, tfrecords_filename_val, batch_size, num_threads)



    print (model_obj.model.summary())
    print('len(model.trainable_variables)', len(model_obj.model.layers))

    parallel_model = multi_gpu_model(model_obj.model, gpus=num_gpus)
    parallel_model.compile(optimizer=tf.keras.optimizers.Adam(lr=learning_rate),
                  loss=my_cost_MSE, metrics=metric_list)


    tensorboard = TensorBoard(log_dir=FolderName+'/logs/'+tensorboard_name)

    checkpoint_path = FolderName + "/tmp/model_epoch{epoch:02d}.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    cp_callback = k.callbacks.ModelCheckpoint(checkpoint_path, verbose=1, period=5, save_weights_only=False)
    callbacks_list = [tensorboard, cp_callback]
    print(parallel_model.summary())
    print('len(model.trainable_variables)', len(parallel_model.layers))
    history = parallel_model.fit(image, annotation, epochs=epoch, batch_size=batch_size, steps_per_epoch=num_batch,
                             validation_data=(image_val,annotation_val), validation_steps=num_batch_val,callbacks=callbacks_list)
    init_op = tf.global_variables_initializer()
    init_l = tf.local_variables_initializer()
    tf.logging.set_verbosity(tf.logging.ERROR)
    with tf.Session() as sess:
        sess.run(init_op)
        sess.run(init_l)

        sk_map = 0
        for bat_v in range(int(2693/batch_size)):
            img_val, anno_val = sess.run([image, annotation])

            pre_val = parallel_model.predict(img_val)

            y_true_val = np.reshape(anno_val, [-1, 26])
            y_hat_val = np.reshape(pre_val, [-1, 26])




if __name__ == '__main__':
    run_test()