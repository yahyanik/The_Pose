from __future__ import division
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
from model_normal import *
import pickle
# import os
import datetime
from Metrics1 import *
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
from Cost_Cross_entropy import *

# Seyed Yahya Nikouei Spring 2019, parallel learning the MNV2 on COCO dataset for human detection and gesture extractoin
# This file is the corrected version of dataset with the combined DATASET file originally built to pass the reg_factor to the mpdel

'''The Parallel model that works with the loss function but non of the metrics'''


def model_cost_function (num_to_reduce, x, y_tr):

    model = MobileNetV2_normal(num_to_reduce=num_to_reduce, is_training=True, input_size=320, input_placeholder=x)
    metric_obj = metric_custom()
    print(model.output.get_shape())
    l = cost(model.output, y_tr)

    me = metric_obj.Distance_parallel(y_tr, model.output)
    accuracy_all = metric_obj.Accu_parallel(model.output, y_tr)
    au_detection = metric_obj.auc_us_parallel(y_tr, model.output)
    au_class = metric_obj.AUC_all_parallel (y_tr, model.output)
    new_acuracy = metric_obj.new_acuracy_parallel(y_tr, model.output)


    return (l, model, me, accuracy_all, au_detection, au_class, metric_obj, new_acuracy)


def make_parallel (fn, num_gpus, num_to_reduce, **kwargs):

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
                out_split_fn = fn(num_to_reduce, **{k: v[i] for k, v in in_splits.items()})
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


def train (epoch = 500, learning_rate = 0.001, regular_fac = 0.001, num_to_reduce=2, imagesize = 320, batch_size = 32, num_threads = 10, num_gpus = 2):

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

    hm_epochs = epoch
    y_tr = tf.placeholder(dtype=tf.float32, shape=[None, 10, 10, 26], name = 'y')
    x = tf.placeholder(dtype=tf.float32, shape=[None, imagesize, imagesize, 3],name='x')

    loss_matrix, model, me_metrix, accuracy_all_metrix, au_detection_metrix, \
        au_class_metrix, metric_obj, new_acuracy_matrix = make_parallel(model_cost_function, num_gpus, num_to_reduce, x=x, y_tr=y_tr)

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

        for epoch in range(6):
            LogWriter = []
            epoch_loss = 0
            epoch_recal = 0
            epoch_perci = 0
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

                summary, _, c,t1,t2,t3,cost_b, new_re_per,pre = sess.run([merged, optimizer,
                                             loss,f1,f2,f3, cost_general, new_acu,model.output], feed_dict={x: img, y_tr: anno})
                epoch_loss += c
                re, per, _, _ = new_re_per
                epoch_recal += re if re!=-1 else 0
                epoch_perci += per if per!=-1 else 0

                if bat%100 == 0:
                    tok = datetime.datetime.now()
                    print('Batch', bat, 'completed out of', num_batch, 'Batch_loss:', c, 'RECAL:', re, 'PER:', per,
                                                    'time for this batch:', (tok - tik))
                    LogWriter.append('Batch '+ str(bat)+ 'completed out of ' + str(num_batch)+ 'Batch_loss: '+ str(c)+ 'RECAL: ' +str(re)+
                                                    'time for this batch:'+str(tok - tik) + '\n')
                    re_pre = np.reshape(pre,(-1,26))
                    re_tr = np.reshape(anno,(-1,26))
                    print ('annotation (', np.min(re_tr[:,:24]), ',', np.max(re_tr[:,:24]), ',', np.min(re_tr[:,24:]), ',', np.max(re_tr[:,24:]), ')',\
                            'prediction (', np.min(re_pre[:,:24]), ',', np.max(re_pre[:,:24]), ',', np.min(re_pre[:,24:]), ',', np.max(re_pre[:,24:]),')')

            print ('\n', '         ###Epoch###', epoch, 'completed out of', hm_epochs, '###Epoch_loss:### ', (epoch_loss/num_batch))
            LogWriter.append('\n'+'         ###Epoch### '+ str(epoch)+ 'completed out of '+str(hm_epochs)+ '###Epoch_loss:### '+str(epoch_loss)+ '\n')
            print ('RECALL for Training:', epoch_recal/num_batch, 'PERCISION for Training:', epoch_perci/num_batch)
            LogWriter.append('RECALL for classification: '+str (epoch_recal/num_batch)+ 'PERCISION for Training:'+str(epoch_perci/num_batch)+ '\n')
            train_writer.add_summary(summary, epoch)

            if epoch%5 == 0:
                img_val, anno_val = sess.run([image_val, annotation_val])
                for bat_v in range(int(2693/batch_size)):
                    summary_v, area_under_curve, means, auc_m, new_re_per = sess.run([merged, auc, D, AU_m, new_acu],
                    feed_dict={x: img_val, y_tr: anno_val})

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
                LogWriter.append('mean landmark AUC: '+ str(auc_m_epoch / int(2693/batch_size))+ '\n')
                LogWriter.append('detection AUC: '+ str(epoch_auc_v / int(2693 / batch_size))+ '\n')
                LogWriter.append('mean distance: '+ str(epoch_distance_v / int(2693 / batch_size))+ '\n')
                LogWriter.append('mean RECAL for bbx: '+ str(epoch_mAP_ditection_v / int(2693 / batch_size))+ '\n')
                LogWriter.append('RECAL: '+ str(RECAL / int(2693 / batch_size))+ '\n')
                LogWriter.append('PERCISION: '+ str(PERCISION / int(2693 / batch_size))+ '\n')
                LogWriter.append('batch_size = '+ str(batch_size)+ 'learning_rate ='+ str(learning_rate)+ 'regular_fac ='+ str(regular_fac)+ '\n')

            f = open('./'+ FolderName+'/tmp/logs.txt','a')
            [f.write(k) for k in LogWriter]
            f.close()


        coord.request_stop()
        coord.join(threads)

if __name__ == '__main__':
    train(imagesize=320, num_threads=10) # in the parallel we are still doing batch of 32