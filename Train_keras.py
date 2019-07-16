from __future__ import division
import tensorflow as tf
import tensorflow.keras as k
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import ModelCheckpoint
from model_normal_keras import *
import time
import os
from cost_keras import *
from Make_parallel import *
from Metrics_keras import *
from tensorflow.keras.utils import multi_gpu_model

# Seyed Yahya Nikouei Summer 2019, parallel learning the MNV2 on COCO dataset for human detection and gesture extractoin
# This one goes with all the _8 files to train a network with just the detection of the object in a window

def train (epoch = 250, learning_rate = 0.001, regular_fac = 0.01, num_to_reduce=32, imagesize = 320, batch_size = 32, num_threads = 10, num_gpus = 2):

    FolderName = './model_normal{}_2.58.{}_1280_{}_0.5_5'.format(batch_size, learning_rate, regular_fac)
    checkpoint_dir = FolderName + "/tmp/model.ckpt"
    tensorboard_name = 'TB_keypoint_keras_{}'.format(int(time.time()))
    file_checkpoint = FolderName + "/keras/model.h5"
    os.makedirs(FolderName+'/keras')
    os.makedirs(FolderName + '/tmp')
    tfrecords_filename = '../The_Pose/tfrecord/DATASET_COCO.tfrecords'
    tfrecords_filename_val = '../The_Pose/tfrecord/DATASET_VAL.tfrecords'

    cp_callback = k.callbacks.ModelCheckpoint(checkpoint_dir, verbose=1, save_freq=2)
    checkpoint = ModelCheckpoint(file_checkpoint, verbose=1)

    num_batch = int(64115 / batch_size)
    num_batch_val = int(2693/batch_size)

    image, annotation = read_image_tf_data(imagesize, tfrecords_filename, batch_size, num_threads)
    image_val, annotation_val = read_image_tf_data(imagesize, tfrecords_filename_val, batch_size, num_threads)

    model_obj = MobileNetV2_normal_keras(num_to_reduce=num_to_reduce, head_is_training=False, regular_fac=regular_fac,
                                         layers_to_fine_tune=155, include_top=False, fireezed_layers=False)

    metric = metric_custom()
    # irre_metric = Out_Metric()

    parallel_model = multi_gpu_model(model_obj.model, gpus=num_gpus)
    parallel_model.compile(optimizer=tf.keras.optimizers.Adam(lr=learning_rate),
                  loss=my_cost_8, metrics=[metric.RECAL_8, metric.PERCISION_8])

    print (parallel_model.summary())
    print("Number of layers in the model: ", len(parallel_model.layers), 'and 155 in the base model')

    tensorboard = TensorBoard(log_dir=FolderName+'/logs/'+tensorboard_name)
    callbacks_list = [tensorboard, checkpoint]

    history = parallel_model.fit(image, annotation, epochs=epoch, batch_size=batch_size, steps_per_epoch=num_batch,
                                 validation_data=(image_val,annotation_val), validation_steps=num_batch_val,callbacks=callbacks_list)
    # print(irre_metric.get_data())
    #
    # # y_pred = parallel_model.predict(image_val, batch_size=batch_size)
    #
    # with open(FolderName+'/tmp/logs.txt', 'wb') as file_pi:
    #     pickle.dump(history.history, file_pi)
























#     new_acu = metric_obj.new_acuracy_single(new_acuracy_matrix)
#     REcal, percision, _, _,_,_,_ = new_acu
#     tf.summary.scalar('PERCISION score', percision)
#     tf.summary.scalar('RECAL score', REcal)
#
#     merged = tf.summary.merge_all()
#     saver = tf.train.Saver(max_to_keep=10)
#
#     gpu_options = tf.GPUOptions(allow_growth=True)
#     config = tf.ConfigProto(gpu_options=gpu_options)
#     config.gpu_options.allow_growth = True
#     init_op = tf.global_variables_initializer()
#     init_l = tf.local_variables_initializer()
#     tf.logging.set_verbosity(tf.logging.ERROR)
#
#     with tf.Session(config=config) as sess:
#
#         sess.run(init_op)
#         sess.run(init_l)
#         coord = tf.train.Coordinator()
#         threads = tf.train.start_queue_runners(coord=coord)
#         train_writer = tf.summary.FileWriter(FolderName+'/train', sess.graph)
#         test_writer = tf.summary.FileWriter(FolderName+'/test')
#         # saver.restore(sess, "./model_832_2.43.320_1280_0.001_1/tmp/model.ckpt-11")
#
#         for epoch in range(hm_epochs):
#         # for epoch in range(1):
#         #     print(epoch)
#             LogWriter = []
#             epoch_loss = 0
#             epoch_recal = 0
#             epoch_perci = 0
#             auc_m_epoch = 0
#             epoch_auc_v = 0
#             train_sk_mPA = 0
#             epoch_distance_v = 0
#             epoch_mAP_ditection_v = 0
#             RECAL = 0
#             PERCISION = 0
#             tik = datetime.datetime.now()
#             ii = 0
#
#             for bat in range(num_batch):
#
#             # for bat in range(1):
#                 img, anno = sess.run([image, annotation])
#                 # ii+=1
#                 summary, _, c,t1,t2,t3,cost_b, new_re_per,pre = sess.run([merged, optimizer,
#                                              loss,f1,f2,f3, cost_general, new_acu,MODEL_OUT], feed_dict={x: img, y_tr: anno})
#                 epoch_loss += c
#                 re, per, mAP, FP_rate,total,ones,summ = new_re_per
#                 epoch_recal += re if re!=-1 else 0
#                 epoch_perci += per if per!=-1 else 0
#                 # print (ii)
#                 y_true_re = np.reshape(anno, [-1, 26])
#                 y_hat_re = np.reshape(pre, [-1, 8])
#
#                 sk_map1 = average_precision_score(np.sign(y_true_re[:, 2]).astype(int), y_hat_re[:, 0])
#                 sk_map2 = average_precision_score(np.sign(y_true_re[:, 5]).astype(int), y_hat_re[:, 1])
#                 sk_map3 = average_precision_score(np.sign(y_true_re[:, 8]).astype(int), y_hat_re[:, 2])
#                 sk_map4 = average_precision_score(np.sign(y_true_re[:, 11]).astype(int), y_hat_re[:, 3])
#                 sk_map5 = average_precision_score(np.sign(y_true_re[:, 14]).astype(int), y_hat_re[:, 4])
#                 sk_map6 = average_precision_score(np.sign(y_true_re[:, 17]).astype(int), y_hat_re[:, 5])
#                 sk_map7 = average_precision_score(np.sign(y_true_re[:, 20]).astype(int), y_hat_re[:, 6])
#                 sk_map8 = average_precision_score(np.sign(y_true_re[:, 21]).astype(int), y_hat_re[:, 7])
#
#                 sk_map = (sk_map1 + sk_map2 + sk_map3 + sk_map4 + sk_map5 + sk_map6 + sk_map7 + sk_map8) / 8.0
#                 if not math.isnan(sk_map):
#                     train_sk_mPA+=sk_map
#                     ii+=1
#
#                 if bat%100 == 0:
#                     tok = datetime.datetime.now()
#
#
#                     print('Batch', bat, 'completed out of', num_batch, 'Batch_loss:', c, 'RECAL:', re, 'PER:', per,
#                                                    'time for this batch:', (tok - tik))
#                     LogWriter.append('Batch '+ str(bat)+ 'completed out of ' + str(num_batch)+ 'Batch_loss: '+ str(c)+ 'RECAL: ' +str(re)+
#                                                     'PER:'+ str(per)+ 'time for this batch:'+str(tok - tik) + '\n')
#                     print ('mAP', mAP, 'FP_rate', FP_rate,total,ones,summ,sk_map)
#                     LogWriter.append('mAP'+ str(mAP)+ 'FP_rate'+ str(FP_rate)+ str(total)+ str(ones)+ str(summ)+ str(sk_map)+'\n')
#                     # re_pre = np.reshape(pre,(-1,8))
#                     # re_tr = np.reshape(anno,(-1,26))
#                     # print ('annotation (', np.min(re_tr[:,:24]), ',', np.max(re_tr[:,:24]), ',', np.min(re_tr[:,24:]), ',', np.max(re_tr[:,24:]), ')',\
#                     #         'prediction (', np.min(re_pre[:,:24]), ',', np.max(re_pre[:,:24]), ',', np.min(re_pre[:,24:]), ',', np.max(re_pre[:,24:]),')')
#
#
#             print ('\n', '         ###Epoch###', epoch, 'completed out of', hm_epochs, '###Epoch_loss:### ', (epoch_loss/num_batch))
#             LogWriter.append('\n'+'         ###Epoch### '+ str(epoch)+ 'completed out of '+str(hm_epochs)+ '###Epoch_loss:### '+str(epoch_loss/num_batch)+ '\n')
#             print ('RECALL for Training:', epoch_recal/num_batch, 'PERCISION for Training:', epoch_perci/num_batch,'sk_mAP:', train_sk_mPA/ii)
#             LogWriter.append('RECALL for classification: '+str (epoch_recal/num_batch)+ 'PERCISION for Training:'+str(epoch_perci/num_batch)+ 'sk_mAP:'+ str(train_sk_mPA/num_batch)+'\n')
#             train_writer.add_summary(summary, epoch)
#
#             if epoch%2 == 0:
#                 img_val, anno_val = sess.run([image_val, annotation_val])
#                 sk_map = 0
#                 for bat_v in range(int(2693/batch_size)):
#                     summary_v, new_re_per, pre_val = sess.run([merged, new_acu, MODEL_OUT],feed_dict={x: img_val, y_tr: anno_val})
#
#                     # epoch_auc_v+=area_under_curve
#                     # a, b = means
#                     epoch_mAP_ditection_v += mAP
#                     # epoch_distance_v += a
#                     # auc_m_epoch += auc_m
#                     re, pe, mAP, FP_rate,_,_,_ = new_re_per
#                     RECAL += re
#                     PERCISION += pe
#
#                     y_true_val = np.reshape(anno_val, [-1, 26])
#                     y_hat_val = np.reshape(pre_val, [-1, 8])
#                     sk_map1 = average_precision_score(np.sign(y_true_val[:, 2]).astype(int), y_hat_val[:, 0])
#                     sk_map2 = average_precision_score(np.sign(y_true_val[:, 5]).astype(int), y_hat_val[:, 1])
#                     sk_map3 = average_precision_score(np.sign(y_true_val[:, 8]).astype(int), y_hat_val[:, 2])
#                     sk_map4 = average_precision_score(np.sign(y_true_val[:, 11]).astype(int), y_hat_val[:, 3])
#                     sk_map5 = average_precision_score(np.sign(y_true_val[:, 14]).astype(int), y_hat_val[:, 4])
#                     sk_map6 = average_precision_score(np.sign(y_true_val[:, 17]).astype(int), y_hat_val[:, 5])
#                     sk_map7 = average_precision_score(np.sign(y_true_val[:, 20]).astype(int), y_hat_val[:, 6])
#                     sk_map8 = average_precision_score(np.sign(y_true_val[:, 21]).astype(int), y_hat_val[:, 7])
#
#                     sk_map += (sk_map1 + sk_map2 + sk_map3 + sk_map4 + sk_map5 + sk_map6 + sk_map7 + sk_map8) / 8.0
#
#                 test_writer.add_summary(summary_v, epoch)
#                 save_path = saver.save(sess, FolderName + "/tmp/model.ckpt", global_step=epoch + 1)
#                 # print('mean landmark AUC:', auc_m_epoch / int(2693/batch_size))
#                 # print('detection AUC:', epoch_auc_v / int(2693 / batch_size))
#                 # print('mean distance:', epoch_distance_v / int(2693 / batch_size))
#                 print('mean RECAL for bbx:', epoch_mAP_ditection_v / int(2693 / batch_size))
#                 print('RECAL:', RECAL / int(2693 / batch_size))
#                 print('PERCISION:', PERCISION / int(2693 / batch_size))
#                 print("Model saved in path: %s" % save_path)
#                 print('batch_size =', batch_size, 'learning_rate =', learning_rate, 'regular_fac =', regular_fac)
#                 print ('sk-mAP test', sk_map/int(2693 / batch_size))
#                 # LogWriter.append('mean landmark AUC: '+ str(auc_m_epoch / int(2693/batch_size))+ '\n')
#                 # LogWriter.append('detection AUC: '+ str(epoch_auc_v / int(2693 / batch_size))+ '\n')
#                 # LogWriter.append('mean distance: '+ str(epoch_distance_v / int(2693 / batch_size))+ '\n')
#                 # LogWriter.append('mean RECAL for bbx: '+ str(epoch_mAP_ditection_v / int(2693 / batch_size))+ '\n')
#                 LogWriter.append('RECAL: '+ str(RECAL / int(2693 / batch_size))+ '\n')
#                 LogWriter.append('PERCISION: '+ str(PERCISION / int(2693 / batch_size))+ '\n')
#                 LogWriter.append('batch_size = '+ str(batch_size)+ 'learning_rate ='+ str(learning_rate)+ 'regular_fac ='+ str(regular_fac)+ '\n')
#                 LogWriter.append('sk-mAP test'+ str(sk_map / int(2693 / batch_size))+'\n')
#
#             f = open(FolderName+'/tmp/logs.txt','a')
#             [f.write(k) for k in LogWriter]
#             f.close()
#
#
#         coord.request_stop()
#         coord.join(threads)
#
if __name__ == '__main__':
    train(imagesize=320, num_threads=10) # in the parallel we are still doing batch of 32