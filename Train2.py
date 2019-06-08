from __future__ import division
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
from model_normal import *
import pickle
# import os
import datetime
from Metrics1 import *
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

# Seyed Yahya Nikouei Spring 2019, parallel learning the MNV2 on COCO dataset for human detection and gesture extractoin

'''The Parallel model that works with the loss function but non of the metrics'''

def CPU_reminder_of_cost(wpar, y_true, batch_size, regular_fac):

    landa = 1
    lada = 0.0755
    gamma = 1          #next change to 5,0.5,1
    bbx_factor = 1

    c1 = tf.reduce_sum(wpar[0], axis=-1)
    c2 = tf.reduce_sum(wpar[1], axis=-1)
    c3 = tf.reduce_sum(wpar[2], axis=-1)
    c4 = tf.reduce_sum(wpar[3], axis=-1)
    c5 = tf.reduce_sum(wpar[4], axis=-1)
    c6 = tf.reduce_sum(wpar[5], axis=-1)
    c7 = tf.reduce_sum(wpar[6], axis=-1)

    v1 = tf.reduce_sum(wpar[7], axis=-1)
    v2 = tf.reduce_sum(wpar[8], axis=-1)
    v3 = tf.reduce_sum(wpar[9], axis=-1)
    v4 = tf.reduce_sum(wpar[10], axis=-1)
    v5 = tf.reduce_sum(wpar[11], axis=-1)
    v6 = tf.reduce_sum(wpar[12], axis=-1)
    v7 = tf.reduce_sum(wpar[13], axis=-1)
    v8 = tf.reduce_sum(wpar[14], axis=-1)

    v11 = tf.reduce_sum(wpar[15], axis=-1)
    v21 = tf.reduce_sum(wpar[16], axis=-1)
    v31 = tf.reduce_sum(wpar[17], axis=-1)
    v41 = tf.reduce_sum(wpar[18], axis=-1)
    v51 = tf.reduce_sum(wpar[19], axis=-1)
    v61 = tf.reduce_sum(wpar[20], axis=-1)
    v71 = tf.reduce_sum(wpar[21], axis=-1)
    v81 = tf.reduce_sum(wpar[22], axis=-1)

    b1 = tf.reduce_sum(wpar[23], axis=-1)
    b2 = tf.reduce_sum(wpar[24], axis=-1)

    f1 = (landa * (c1 + c2 + c3 + c4 + c5 + c6 + c7 + b1))
    f2 = (gamma * (v11+v21+v31+v41+v51+v61+v71+v81))
    f3 = (lada * (v1+v2+v3+v4+v5+v6+v7+v8))
    f4 = (bbx_factor * b2)
    cost_general = (f1+f2+f3+f4)

    # vars_all = tf.trainable_variables()
    # L = 0
    # for v in vars_all:
    #         print('var shape', v.get_shape())
    #         L += tf.nn.l2_loss([tf.cast(v, tf.float32)])* regular_fac
    # lossL2 =l2_loss = weight_decay * tf.add_n(
    l2_loss = regular_fac * tf.add_n([tf.nn.l2_loss(tf.cast(v, tf.float32))
                                      for v in tf.trainable_variables() if 'bias' not in v.name])
    # reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    # cost_total = (cost_general + (regular_fac/2) * sum(reg_losses)) / batch_size
    cost_total = (cost_general+l2_loss/2)/batch_size

    return cost_total,f1,f2,f3,cost_general


def cost (y_pred, y_true):

    y = tf.reshape(y_true, [-1, 26])
    y_hat = tf.reshape(y_pred, [-1, 26])

    c1 = tf.sign(y[:,2]) * (
            tf.squared_difference(y[:,0] , y_hat[:,0]) +
            tf.squared_difference(y[:,1] , y_hat[:,1]))

    c2 = tf.sign(y[:, 5]) * (
            tf.squared_difference(y[:, 3] , y_hat[:, 3]) +
            tf.squared_difference(y[:, 4] , y_hat[:, 4]))

    c3 = tf.sign(y[:, 8]) * (
            tf.squared_difference(y[:, 6] , y_hat[:, 6]) +
            tf.squared_difference(y[:, 7] , y_hat[:, 7]))

    c4 = tf.sign(y[:, 11]) * (
            tf.squared_difference(y[:, 10] , y_hat[:, 10]) +
            tf.squared_difference(y[:, 9] , y_hat[:, 9]))

    c5 = tf.sign(y[:, 14]) * (
            tf.squared_difference(y[:, 12] , y_hat[:, 12]) +
            tf.squared_difference(y[:, 13] , y_hat[:, 13]))

    c6 = tf.sign(y[:, 17]) * (
            tf.squared_difference(y[:, 15] , y_hat[:, 15]) +
            tf.squared_difference(y[:, 16] , y_hat[:, 16]))

    c7 = tf.sign(y[:, 20]) * (
            tf.squared_difference(y[:, 18] , y_hat[:, 18]) +
            tf.squared_difference(y[:, 19] , y_hat[:, 19]))

    v1 = ((-1) * tf.sign(y[:, 2]) + 1) * (tf.squared_difference(y_hat[:, 2] , tf.sign(y[:, 2])))
    v2 = ((-1) * tf.sign(y[:, 5]) + 1) * (tf.squared_difference(y_hat[:, 5] , tf.sign(y[:, 5])))
    v3 = ((-1) * tf.sign(y[:, 8]) + 1) * (tf.squared_difference(y_hat[:, 8] , tf.sign(y[:, 8])))
    v4 = ((-1) * tf.sign(y[:, 11]) + 1) * (tf.squared_difference(y_hat[:, 11] , tf.sign(y[:, 11])))
    v5 = ((-1) * tf.sign(y[:, 14]) + 1) * (tf.squared_difference(y_hat[:, 14] , tf.sign(y[:, 14])))
    v6 = ((-1) * tf.sign(y[:, 17]) + 1) * (tf.squared_difference(y_hat[:, 17] , tf.sign(y[:, 17])))
    v7 = ((-1) * tf.sign(y[:, 20]) + 1) * (tf.squared_difference(y_hat[:, 20] , tf.sign(y[:, 20])))
    v8 = ((-1) * tf.sign(y[:, 21]) + 1) * (tf.squared_difference(y_hat[:, 21] , tf.sign(y[:, 21])))

    v11 = tf.sign(y[:, 2]) *(tf.squared_difference(y_hat[:, 2] , tf.sign(y[:, 2])))       # in model_1280relu_7 series there is no sign() for the
    v21 = tf.sign(y[:, 5]) *(tf.squared_difference(y_hat[:, 5] , tf.sign(y[:, 5])))   # y_hat in the error calculation just one for before the error
    v31 = tf.sign(y[:, 8]) *(tf.squared_difference(y_hat[:, 8] , tf.sign(y[:, 8])))
    v41 = tf.sign(y[:, 11]) *(tf.squared_difference(y_hat[:, 11] , tf.sign(y[:, 11])))
    v51 = tf.sign(y[:, 14]) *(tf.squared_difference(y_hat[:, 14] , tf.sign(y[:, 14])))
    v61 = tf.sign(y[:, 17]) *(tf.squared_difference(y_hat[:, 17] , tf.sign(y[:, 17])))
    v71 = tf.sign(y[:, 20]) *(tf.squared_difference(y_hat[:, 20] , tf.sign(y[:, 20])))
    v81 = tf.sign(y[:, 21]) *(tf.squared_difference(y_hat[:, 21] , tf.sign(y[:, 21])))

    b1 = tf.sign(y[:, 21]) * (
            tf.squared_difference(y[:, 22] , y_hat[:, 22]) +
            tf.squared_difference(y[:, 23] , y_hat[:, 23]))

    b2 = tf.sign(y[:, 21]) * (
            tf.squared_difference(tf.sqrt(y[:, 24]) , tf.sqrt(y_hat[:, 24])) +
            tf.squared_difference(tf.sqrt(y[:, 25]) , tf.sqrt(y_hat[:, 25])))

    cost = (c1,c2,c3,c4,c5,c6,c7,v1,v2,v3,v4,v5,v6,v7,v8,v11,v21,v31,v41,v51,v61,v71,v81,b1,b2)
    return cost


def model_cost_function (x, y_tr):

    model = MobileNetV2_normal(num_to_reduce=2, is_training=True, input_size=320, input_placeholder=x)
    metric_obj = metric_custom()
    print(model.output.get_shape())
    l = cost(model.output, y_tr)

    me = metric_obj.Distance_parallel(y_tr, model.output)
    accuracy_all = metric_obj.Accu_parallel(model.output, y_tr)
    au_detection = metric_obj.auc_us_parallel(y_tr, model.output)
    au_class = metric_obj.AUC_all_parallel (y_tr, model.output)
    new_acuracy = metric_obj.new_acuracy_parallel(y_tr, model.output)


    return (l, model, me, accuracy_all, au_detection, au_class, metric_obj, new_acuracy)


def make_parallel (fn, num_gpus, **kwargs):

    in_splits = {}

    for k, v in kwargs.items():
        in_splits[k] = tf.split(v, num_gpus)

    out_split_l = []
    out_split_me = []
    out_split_accuracy_all = []
    out_split_au_detection = []
    out_split_au_class = []
    out_aplit_new_acuracy = []
    for i in range(num_gpus):
        with tf.device(tf.DeviceSpec(device_type="GPU", device_index=i)):
            with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
                out_split_fn = fn(**{k: v[i] for k, v in in_splits.items()})
                out_split_l.append(out_split_fn[0]) # [0] so that only lost is considered not the model
                out_split_me.append(out_split_fn[2])
                out_split_accuracy_all.append(out_split_fn[3])
                out_split_au_detection.append(out_split_fn[4])
                out_split_au_class.append(out_split_fn[5])
                model = (out_split_fn) [1]
                metric_obj = (out_split_fn)[6]
                out_aplit_new_acuracy.append(out_split_fn[7])

    list_metrix = []
    me_metrix = []
    accuracy_all_metrix = []
    au_detection_metrix= []
    au_class_metrix= []
    new_acuracy_matrix = []

    for wpar in (range(len(out_split_l[0]))):
        list_metrix.append(tf.concat([out_split_l[z][wpar] for z in range(len(out_split_l))], axis=0))
    for wpar in (range(len(out_split_me[0]))):
        me_metrix.append(tf.concat([out_split_me[z][wpar] for z in range(len(out_split_me))], axis=0))
    for wpar in (range(len(out_split_accuracy_all[0]))):
        accuracy_all_metrix.append(tf.concat([out_split_accuracy_all[z][wpar] for z in range(len(out_split_accuracy_all))], axis=0))
    for wpar in (range(len(out_split_au_detection[0]))):
        au_detection_metrix.append(tf.concat([out_split_au_detection[z][wpar] for z in range(len(out_split_au_detection))], axis=0))
    for wpar in (range(len(out_split_au_class[0]))):
        au_class_metrix.append(tf.concat([out_split_au_class[z][wpar] for z in range(len(out_split_au_class))], axis=0))
    for wpar in (range(len(out_aplit_new_acuracy[0]))):
        new_acuracy_matrix.append(tf.concat([out_aplit_new_acuracy[z][wpar] for z in range(len(out_aplit_new_acuracy))], axis=0))

    return list_metrix, model, me_metrix, accuracy_all_metrix, au_detection_metrix, au_class_metrix, metric_obj, new_acuracy_matrix


def train (imagesize = 320, batch_size = 16, num_threads = 10, epoch = 500, learning_rate = 0.01, regular_fac = 0.001, num_gpus = 2):
    #TODO: use the regularization of the weights - add the mpii
    FolderName = './model_normalSig{}_2.26.320_52_{}'.format(batch_size, regular_fac)
    num_batch = int(64115 / batch_size)     #64114 is the number of examples in the dataset
    tfrecords_filename = '../The_Pose/tfrecord/keypoint_train_Corrected.1.tfrecords'
    # tfrecords_filename_val = '../The_Pose/tfrecord/keypoint_val_Corrected.1.tfrecords'
    # tfrecords_filename_mpii = '../The_Pose/tfrecord/keypoint_train_MPII.1.tfrecords'
    filename_queue = tf.train.string_input_producer([tfrecords_filename])
    # filename_queue_val = tf.train.string_input_producer([tfrecords_filename_val])
    # filename_queue_mpii = tf.train.string_input_producer([tfrecords_filename_mpii])
    image, annotation = read_and_decode(imagesize, filename_queue, batch_size, num_threads)
    # image_val, annotation_val = read_and_decode(imagesize, filename_queue_val, batch_size, num_threads)
    # image_mpii, annotation_mpii = read_and_decode(imagesize, filename_queue_mpii, batch_size, num_threads)

    with open('../The_Pose/tfrecord/X_Corrected_val_3.1.pickle', 'rb') as f:
        X_v = pickle.load(f) / 255
    X_val = np.array(X_v)
    with open('../The_Pose/tfrecord/y_Corrected_val_3.1.pickle', 'rb') as f:
        y_v = pickle.load(f)
    y_val = np.array(y_v)

    hm_epochs = epoch
    y_tr = tf.placeholder(dtype=tf.float32, shape=[None, 10, 10, 26], name = 'y')
    x = tf.placeholder(dtype=tf.float32, shape=[None, imagesize, imagesize, 3],name='x')

    loss_matrix, model, me_metrix, accuracy_all_metrix, au_detection_metrix, \
        au_class_metrix, metric_obj, new_acuracy_matrix = make_parallel(model_cost_function, num_gpus, x=x, y_tr=y_tr)

    loss,f1,f2,f3, cost_general = CPU_reminder_of_cost(loss_matrix, y_tr, batch_size, regular_fac)
    tf.summary.scalar('loss', loss)
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(loss, colocate_gradients_with_ops=True)

    auc = metric_obj.auc_us_single(au_detection_metrix)
    tf.summary.scalar('AUC for human localization', auc)
    AU_m = metric_obj.AUC_all_single(au_class_metrix)
    tf.summary.scalar('AUC_mean_keypoints', AU_m)
    D = metric_obj.Distance_single(me_metrix, y_tr)
    a, b = D
    tf.summary.scalar('average distance from target', a)
    tf.summary.scalar('mAP for human detection', b)
    new_acu = metric_obj.new_acuracy_single(new_acuracy_matrix)
    REcal, percision, _, _ = new_acu
    tf.summary.scalar('PERCISION score', percision)
    tf.summary.scalar('RECAL score', REcal)

    merged = tf.summary.merge_all()
    saver = tf.train.Saver(max_to_keep=10)

    gpu_options = tf.GPUOptions(allow_growth=True)
    config = tf.ConfigProto(gpu_options=gpu_options)
    config.gpu_options.allow_growth = True
    init_op = tf.global_variables_initializer()
    init_l = tf.local_variables_initializer()
    tf.logging.set_verbosity(tf.logging.ERROR)

    with tf.Session(config=config) as sess:

        sess.run(init_op)
        sess.run(init_l)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        train_writer = tf.summary.FileWriter(FolderName+'/train', sess.graph)
        test_writer = tf.summary.FileWriter(FolderName+'/test')
        # saver.restore(sess, "./model_normal_2.2.1280_676/tmp/model.ckpt-6")

        for epoch in range(hm_epochs):
            epoch_loss = 0
            epoch_recal = 0
            auc_m_epoch = 0
            epoch_auc_v = 0
            epoch_distance_v = 0
            epoch_mAP_ditection_v = 0
            RECAL = 0
            PERCISION = 0
            tik = datetime.datetime.now()

            for bat in range(num_batch):
            # for bat in range(1):
                img, anno = sess.run([image, annotation])
                # img_mpii, anno_mpii = sess.run([image_mpii, annotation_mpii])

                summary, _, c,_,_,_,cost_b, new_re_per,pre = sess.run([merged, optimizer,
                                             loss,f1,f2,f3, cost_general, new_acu,model.output], feed_dict={x: img, y_tr: anno})
                epoch_loss += c
                re, _, _, _ = new_re_per
                epoch_recal += re

                if bat%100 == 0:
                    tok = datetime.datetime.now()
                    print('Batch', bat, 'completed out of', num_batch, 'Batch_loss:', c, 'RECAL:', re,
                                                    'time for this batch:', c, cost_b, (tok - tik))
                    re_pre = np.reshape(pre,(-1,26))
                    re_tr = np.reshape(anno,(-1,26))
                    print ('annotation (', np.min(re_tr[:,:24]), ',', np.max(re_tr[:,:24]), ',', np.min(re_tr[:,24:]), ',', np.max(re_tr[:,24:]), ')',\
                            'prediction (', np.min(re_pre[:,:24]), ',', np.max(re_pre[:,:24]), ',', np.min(re_pre[:,24:]), ',', np.max(re_pre[:,24:]),')')

            print ('         ###Epoch###', epoch, 'completed out of', hm_epochs, '###Epoch_loss:### ', epoch_loss)
            print ('RECALL for classification:', epoch_recal/num_batch)
            train_writer.add_summary(summary, epoch)

            if epoch%5 == 0:
                # img_val, anno_val = sess.run([image_val, annotation_val])
                for bat_v in range(int(2693/batch_size)):
                    summary_v, area_under_curve, means, auc_m, new_re_per = sess.run([merged, auc, D, AU_m, new_acu],
                    feed_dict={x: X_val[bat_v*batch_size:(bat_v+1)*batch_size], y_tr: y_val[bat_v*batch_size:(bat_v+1)*batch_size]})
                    # summary_v, area_under_curve, means, auc_m, new_re_per = sess.run([merged, auc, D, AU_m, new_acu],
                    # feed_dict={x: img_val, y_tr: anno_val})

                    epoch_auc_v+=area_under_curve
                    a, b = means
                    epoch_mAP_ditection_v += b
                    epoch_distance_v += a
                    auc_m_epoch += auc_m
                    re, pe, _, _ = new_re_per
                    RECAL += re
                    PERCISION += pe

                test_writer.add_summary(summary_v, epoch)
                save_path = saver.save(sess, FolderName + "/tmp/model.ckpt", global_step=epoch + 1)
                print('mean landmark AUC:', auc_m_epoch / int(2693/batch_size))
                print('detection AUC:', epoch_auc_v / int(2693 / batch_size))
                print('mean distance:', epoch_distance_v / int(2693 / batch_size))
                print('mean RECAL for bbx:', epoch_mAP_ditection_v / int(2693 / batch_size))
                print('RECAL:', RECAL / int(2693 / batch_size))
                print('PERCISION:', PERCISION / int(2693 / batch_size))
                print("Model saved in path: %s" % save_path)
                print('batch_size =', batch_size, 'learning_rate =', learning_rate, 'regular_fac =', regular_fac)

        coord.request_stop()
        coord.join(threads)

if __name__ == '__main__':
    train(imagesize=320, num_threads=10) # in the parallel we are still doing batch of 32