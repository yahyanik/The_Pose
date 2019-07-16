from __future__ import division
import tensorflow as tf
import numpy as np
import skimage.io as io


'''
These functions are the metrics that are needed for performance monitoring. The functions are known with their names and 
tailored to the spesific dataset made to train Human Gestures.
'''

def read_and_decode(imagesize, filename_queue, batch_size, num_threads):


    reader = tf.TFRecordReader()

    _, serialized_example = reader.read(filename_queue)
    # print serialized_example.shape

    features = tf.parse_single_example(
      serialized_example,
      # Defaults are not specified since both keys are required.
      features={
        'height': tf.FixedLenFeature([], tf.int64),
        'width': tf.FixedLenFeature([], tf.int64),
        'image_raw': tf.FixedLenFeature([], tf.string),
        'mask_raw': tf.FixedLenFeature([], tf.string)
        })

    image = tf.decode_raw(features['image_raw'], tf.uint8)
    annotation = tf.decode_raw(features['mask_raw'], tf.float16)

    height = tf.cast(features['height'], tf.int32)
    # print height
    width = tf.cast(features['width'], tf.int32)

    image_shape = tf.stack([height, width, 3])
    annotation_shape = tf.stack([10, 10, 26])
    annotation = tf.reshape(annotation, annotation_shape)

    # annotation = tf.reshape(annotation, [-1])


    image = tf.reshape(image, image_shape)

    # print image.shape
    # normalize_factor = tf.constant((255), name = 'normalize_factor')
    image = image /255
    # annotation = tf.reshape(annotation, annotation_shape)
    # annotation = tf.reshape(annotation, annotation_shape)


    # IMAGE_HEIGHT = tf.constant((imagesize, imagesize, 3), dtype=tf.int32)
    # annotation_size_const = tf.constant((IMAGE_size, IMAGE_size, 1), dtype=tf.int32)

    # Random transformations can be put here: right before you crop images
    # to predefined size. To get more information look at the stackoverflow
    # question linked above.

    resized_image = tf.image.resize_image_with_crop_or_pad(image=image, target_height=imagesize, target_width=imagesize)    #redundant in case of change later

    # resized_image = resized_image / 255
    # resized_annotation = tf.image.resize_image_with_crop_or_pad(image=annotation, target_height=IMAGE_HEIGHT,target_width=IMAGE_WIDTH)

    # print resized_image.shape

    # dataset1 = tf.data.Dataset.from_tensor_slices([resized_image, annotation])
    #
    # dataset1 = dataset1.batch(32, drop_remainder=False)
    #
    # images, annotations = dataset1
    # print images.shape
    # print annotations.shape

    images, annotations = tf.train.batch( [resized_image, annotation],
                                                 batch_size= batch_size,
                                                 enqueue_many=False,
                                                 capacity=500,
                                                 num_threads= num_threads)

    # images, annotations = tf.data.Dataset.shuffle(min_after_dequeue=1000).batch(batch_size)


    return images, annotations
    # return  resized_image, annotation

# def AP (y_hat, y_true):
#
#     # y_hat = np.reshape(y_hat, [-1, 26])
#     # y_true = np.reshape(y_true, [-1, 26])
#
#     TP = 0
#     FP = 0
#     FN = 0
#
#     TP += tf.reduce_sum(tf.cast(tf.logical_and(y_hat[:,2], y_true[:,2]),'float'))
#     TP += tf.reduce_sum(tf.cast(tf.logical_and(y_hat[:,5], y_true[:,5]),'float'))
#     TP += tf.reduce_sum(tf.cast(tf.logical_and(y_hat[:,8], y_true[:,8]),'float'))
#     TP += tf.reduce_sum(tf.cast(tf.logical_and(y_hat[:,11], y_true[:,11]),'float'))
#     TP += tf.reduce_sum(tf.cast(tf.logical_and(y_hat[:,14], y_true[:,14]),'float'))
#     TP += tf.reduce_sum(tf.cast(tf.logical_and(y_hat[:,17], y_true[:,17]),'float'))
#     TP += tf.reduce_sum(tf.cast(tf.logical_and(y_hat[:,20], y_true[:,20]),'float'))
#     TP += tf.reduce_sum(tf.cast(tf.logical_and(y_hat[:, 21], y_true[:, 21]),'float'))
#
#     FP += tf.reduce_sum(tf.cast(tf.logical_and(y_hat[:,2], tf.logical_not(y_true[:,2])),'float'))
#     FP += tf.reduce_sum(tf.cast(tf.logical_and(y_hat[:,5], tf.logical_not(y_true[:,5])),'float'))
#     FP += tf.reduce_sum(tf.cast(tf.logical_and(y_hat[:,8], tf.logical_not(y_true[:,8])),'float'))
#     FP += tf.reduce_sum(tf.cast(tf.logical_and(y_hat[:,11], tf.logical_not(y_true[:,11])),'float'))
#     FP += tf.reduce_sum(tf.cast(tf.logical_and(y_hat[:,14], tf.logical_not(y_true[:,14])),'float'))
#     FP += tf.reduce_sum(tf.cast(tf.logical_and(y_hat[:,17], tf.logical_not(y_true[:,17])),'float'))
#     FP += tf.reduce_sum(tf.cast(tf.logical_and(y_hat[:,20], tf.logical_not(y_true[:,20])),'float'))
#     FP += tf.reduce_sum(tf.cast(tf.logical_and(y_hat[:,21], tf.logical_not(y_true[:,21])),'float'))
#
#     FN += tf.reduce_sum(tf.cast(tf.logical_and(tf.logical_not(y_hat[:,2]), y_true[:,2]),'float'))
#     FN += tf.reduce_sum(tf.cast(tf.logical_and(tf.logical_not(y_hat[:,5]), y_true[:,5]),'float'))
#     FN += tf.reduce_sum(tf.cast(tf.logical_and(tf.logical_not(y_hat[:,8]), y_true[:,8]),'float'))
#     FN += tf.reduce_sum(tf.cast(tf.logical_and(tf.logical_not(y_hat[:,11]), y_true[:,11]),'float'))
#     FN += tf.reduce_sum(tf.cast(tf.logical_and(tf.logical_not(y_hat[:,14]), y_true[:,14]),'float'))
#     FN += tf.reduce_sum(tf.cast(tf.logical_and(tf.logical_not(y_hat[:,17]), y_true[:,17]),'float'))
#     FN += tf.reduce_sum(tf.cast(tf.logical_and(tf.logical_not(y_hat[:,20]), y_true[:,20]),'float'))
#     FN += tf.reduce_sum(tf.cast(tf.logical_and(tf.logical_not(y_hat[:,21]), y_true[:,21]),'float'))
#
#     percision = (TP) / (TP + FP)
#     recall = (TP) / (TP + FN)   #as the treshold gets bigger the recall gets smaller
#
#
#
#     # predictions = [y_hat[:,2], y_hat[:,5], y_hat[:,8], y_hat[:,11], y_hat[:,14], y_hat[:,17], y_hat[:,20], y_hat[:,21]]
#     # labels = [y_true[:,2], y_true[:,5], y_true[:,8], y_true[:,11], y_true[:,14], y_true[:,17], y_true[:,20], y_true[:,21]]
#     # y_hat = tf.reshape(y_hat, [-1, 26])
#     # predictions = y_hat [:,2]
#     # y_true = tf.reshape(y_true, [-1, 26])
#     # labels = tf.cast(y_true [:, 2], tf.int64)
#     #
#     # map = tf.metrics.average_precision_at_k(y_true[:,2],tf.cast(y_hat[:,2], tf.float64),k = 1)
#
#     return (percision, recall)

# def miou(y_hat, y_true):
#     per = 0
#     y_hat = tf.reshape(y_hat, [-1, 26])
#     y_true = tf.reshape(y_true, [-1, 26])
#     for k in range (0,11):
#
#         y_hat_tmp = tf.greater(y_hat, k/10)
#         y_true_tmp = tf.greater(y_true, k/10)
#
#         percision, recall = AP(y_hat_tmp, y_true_tmp)
#         # print recall
#         percision = tf.where(tf.is_nan(percision), 0., percision)   # as the treshold gets bigger, the recall gets smaller
#         per+=percision
#
#     return (per/10)




class metric_custom(object):

    def mAP_RealImage (self,Y_true, Y_pred, confidance = 0.5, iou_score=0.5, iou_same_person_score = 0.6):
        with tf.variable_scope('mAP_RealImage '):
            bbx_pred = []                   #extracting the bounding boxes from prediction
            bbx_true = []                   #extracting the bounding boxes from ground truth
            score = 0       # to count how many we detected more than 0.5 correctly
            for k in range (Y_pred.shape[0]):
                for i in range (Y_pred.shape[1]):
                    for j in range (Y_pred.shape[2]):
                        if np.sign(np.maximum((Y_pred[k,i,j,21] - confidance),0)):

                            x_pred = (32 * Y_pred[k,i,j,22] + i*32)
                            y_pred = (32 * Y_pred[k,i,j,22] + j*32)
                            w_pred = (Y_pred[k,i,j,24]*32)
                            h_pred = (Y_pred[k,i,j,25]*32)
                            bbx_pred.append((x_pred, y_pred, w_pred, h_pred))

                for i in range (Y_true.shape[1]):
                    for j in range (Y_true.shape[2]):
                        if Y_true[k,i,j,21] == 1:

                            x_true = (32 * Y_true[k, i, j, 22] + i * 32)
                            y_true = (32 * Y_true[k, i, j, 22] + j * 32)
                            w_true = (Y_true[k, i, j, 24] * 32)
                            h_true = (Y_true[k, i, j, 25] * 32)
                            bbx_true.append((x_true, y_true, w_true, h_true))
                marked = []
                bbx_pred_s = []
                lengh = len(bbx_pred)
                for b in range(lengh):              # calculating the iou score for the y_pred to make the ones that represent the same object deleted
                    for v in range(b+1, lengh):

                        xA = np.maximum((bbx_pred[v][0] - bbx_pred[v][2] / 2), (bbx_pred[b][0] - bbx_pred[b][2] / 2))
                        yA = np.maximum((bbx_pred[v][1] - bbx_pred[v][3] / 2), (bbx_pred[b][1] - bbx_pred[b][3] / 2))
                        xB = np.minimum((bbx_pred[v][0] + bbx_pred[v][2] / 2), (bbx_pred[b][0] + bbx_pred[b][2] / 2))
                        yB = np.minimum((bbx_pred[v][1] + bbx_pred[v][3] / 2), (bbx_pred[b][1] + bbx_pred[b][3] / 2))
                        interArea = (np.maximum(0.0, xB - xA) * np.maximum(0.0, yB - yA))
                        boxAArea = (bbx_pred[b][2]) * (bbx_pred[b][3])
                        boxBArea = (bbx_pred[v][2]) * (bbx_pred[v][3])
                        iou1 = interArea / (boxAArea + boxBArea - interArea)
                        if iou1 >= iou_same_person_score:
                            marked.append(b)
                            break

                for bbx in range(len(bbx_pred)):
                    if bbx not in marked:
                        bbx_pred_s.append(bbx_pred[bbx])



                for b in bbx_pred:              # calculating the iou score
                    for v in bbx_true:

                        xA = np.maximum((v[0] - v[2] / 2), (b[0] - b[2] / 2))
                        yA = np.maximum((v[1] - v[3] / 2), (b[1] - b[3] / 2))
                        xB = np.minimum((v[0] + v[2] / 2), (b[0] + b[2] / 2))
                        yB = np.minimum((v[1] + v[3] / 2), (b[1] + b[3] / 2))
                        interArea = (np.maximum(0.0, xB - xA) * np.maximum(0.0, yB - yA))
                        boxAArea = (b[2]) * (b[3])
                        boxBArea = (v[2]) * (v[3])
                        iou = interArea / (boxAArea + boxBArea - interArea)

                        if iou >= iou_score:
                            score += 1

            mAP = score/(np.count_nonzero(Y_true[:,:,:,21]))
            # mAP = np.maximum(mAP, 0.0)
        return (mAP, bbx_pred)


    def Distance_single (self,wpar, y_true, iou_confidance=0.5):
        with tf.variable_scope('Distance_single'):
            y_hat = wpar[7]
            y = tf.cast(tf.reshape(y_true, [-1, 26]), tf.float32)
            c1 = tf.reduce_sum(wpar[0])
            c2 = tf.reduce_sum(wpar[1])
            c3 = tf.reduce_sum(wpar[2])
            c4 = tf.reduce_sum(wpar[3])
            c5 = tf.reduce_sum(wpar[4])
            c6 = tf.reduce_sum(wpar[5])
            c7 = tf.reduce_sum(wpar[6])

            count1 = tf.cast(tf.count_nonzero(tf.sign(y[:, 2])), 'float')
            count2 = tf.cast(tf.count_nonzero(tf.sign(y[:, 5])), 'float')
            count3 = tf.cast(tf.count_nonzero(tf.sign(y[:, 8])), 'float')
            count4 = tf.cast(tf.count_nonzero(tf.sign(y[:, 11])), 'float')
            count5 = tf.cast(tf.count_nonzero(tf.sign(y[:, 14])), 'float')
            count6 = tf.cast(tf.count_nonzero(tf.sign(y[:, 17])), 'float')
            count7 = tf.cast(tf.count_nonzero(tf.sign(y[:, 20])), 'float')
            summ = c1 + c2 + c3 + c4 + c5 + c6 + c7
            count = count1 + count2 + count3 + count4 + count5 + count6 + count7
            mean_distance = tf.where(tf.logical_not(tf.equal(count, 0)), x=(summ / tf.cast(count, 'float')), y=-1)

            count8 = tf.cast(tf.count_nonzero(y[:, 21]), 'float')
            xA = tf.maximum((y[:, 22] - y[:, 24] / 2), (y_hat[:, 22] - y_hat[:, 24] / 2))
            yA = tf.maximum(y[:, 23] - y[:, 25] / 2, y_hat[:, 23] - y_hat[:, 25] / 2)
            xB = tf.minimum(y[:, 22] + y[:, 24] / 2, y_hat[:, 22] + y_hat[:, 24] / 2)
            yB = tf.minimum(y[:, 23] + y[:, 25] / 2, y_hat[:, 23] + y_hat[:, 25] / 2)
            interArea = tf.sign(y[:, 21]) * (tf.maximum(0.0, xB - xA) * tf.maximum(0.0, yB - yA))
            boxAArea = tf.sign(y[:, 21]) * ((y_hat[:, 24]) * (y_hat[:, 25]))
            boxBArea = tf.sign(y[:, 21]) * ((y[:, 24]) * (y[:, 25]))
            iou = tf.abs(interArea / (boxAArea + boxBArea - interArea))
            kk = tf.reduce_sum(tf.cast(tf.greater(iou, iou_confidance), 'float'))
            mAP_at_iou_half = (kk) / count8
        return (mean_distance, mAP_at_iou_half)

    def Distance_parallel (self,y_true, y_pred):
        with tf.variable_scope('Distance_parallel'):

            y_hat = tf.reshape(y_pred, [-1, 26])
            y = tf.cast(tf.reshape(y_true, [-1, 26]), tf.float32)

            c1 = tf.sqrt(tf.sign(y[:, 2]) *
                               (tf.keras.backend.square(y[:, 0] - y_hat[:, 0]) + tf.keras.backend.square(y[:, 1] - y_hat[:, 1])))
    # count1 = tf.cast(tf.count_nonzero(tf.sign(y[:, 2])), 'float')
    # mean1 = tf.where(tf.equal(count1, 0.0), x = (c1 / count1), y = 0)

            c2 = tf.sqrt(tf.sign(y[:, 5]) *
                               (tf.keras.backend.square(y[:, 3] - y_hat[:, 3]) + tf.keras.backend.square(y[:, 4] - y_hat[:, 4])))
    # count2 = tf.cast(tf.count_nonzero(y[:, 5]), 'float')
    # mean2 = tf.where(tf.equal(count2, 0.0), x = (c2 / count2), y = 0)

            c3 = tf.sqrt(tf.sign(y[:, 8]) *
                               (tf.keras.backend.square(y[:, 6] - y_hat[:, 6]) + tf.keras.backend.square(y[:, 7] - y_hat[:, 7])))
    # count3 = tf.cast(tf.count_nonzero(tf.sign(y[:, 8])), 'float')
    # mean3 = tf.where(tf.equal(count3, 0.0), x = (c3 / count3), y = 0)

            c4 = tf.sqrt(tf.sign(y[:, 11]) *
                               (tf.keras.backend.square(y[:, 9] - y_hat[:, 9]) + tf.keras.backend.square(y[:, 10] - y_hat[:, 10])))
    # count4 = tf.cast(tf.count_nonzero(tf.sign(y[:, 11])), 'float')
    # mean4 = tf.where(tf.equal(count4, 0.0), x = (c4 / count4), y = 0)

            c5 = tf.sqrt(tf.sign(y[:, 14]) *
                               (tf.keras.backend.square(y[:, 12] - y_hat[:, 12]) + tf.keras.backend.square(y[:, 13] - y_hat[:, 13])))
    # count5 = tf.cast(tf.count_nonzero(tf.sign(y[:, 14])), 'float')
    # mean5 = tf.where(tf.equal(count5, 0.0), x = (c5 / count5), y = 0)

            c6 = tf.sqrt(tf.sign(y[:, 17]) *
                               (tf.keras.backend.square(y[:, 15] - y_hat[:, 15]) + tf.keras.backend.square(y[:, 16] - y_hat[:, 16])))
    # count6 = tf.cast(tf.count_nonzero(tf.sign(y[:, 17])), 'float')
    # mean6 = tf.where(tf.equal(count6, 0.0), x = (c6 / count6), y = 0)

            c7 = tf.sqrt(tf.sign(y[:, 20]) *
                               (tf.keras.backend.square(y[:, 18] - y_hat[:, 18]) + tf.keras.backend.square(y[:, 19] - y_hat[:, 19])))
    # count7 = tf.cast(tf.count_nonzero(tf.sign(y[:, 20])), 'float')
    # mean7 = tf.where(tf.equal(count7, 0.0), x = (c7 / count7), y = 0)

    # counter = tf.sign(mean1)+tf.sign(mean2)+tf.sign(mean3)+tf.sign(mean4)+tf.sign(mean5)+tf.sign(mean6)+tf.sign(mean7)

    # mean_distance = (mean1+mean2+mean3+mean4+mean5+mean6+mean7) / counter

            wpar = [c1,c2,c3,c4,c5,c6,c7,y_hat]
    # mean_distance = sum
        return wpar


    def accu_single(self,wpar, y_true):
        with tf.variable_scope('accu_single'):
            y_true = tf.reshape(y_true, [-1, 26])
            summ = tf.reduce_sum(wpar[0]) + tf.reduce_sum(wpar[1]) + tf.reduce_sum(wpar[2]) + tf.reduce_sum(wpar[3]) + \
                tf.reduce_sum(wpar[4]) + tf.reduce_sum(wpar[5]) + tf.reduce_sum(wpar[6])
            list = [2,5,8,11,14,17,20]
            count_keypoints =0
            for i in list:
                count_keypoints += tf.cast(tf.count_nonzero(y_true[:, i]), 'float')
            ac = tf.where(tf.logical_not(tf.equal(count_keypoints, 0)), x=(summ / count_keypoints), y=-1)
        # ac = summ
        return ac

    def Accu_parallel(self,y_hat, y_true, confidance=0.5):
        with tf.variable_scope('Accu_parallel'):

            y_hat = tf.reshape(y_hat, [-1, 26])
            y_true = tf.reshape(y_true, [-1, 26])

            s1 = tf.cast(tf.logical_and(tf.greater(y_hat[:, 2], confidance), tf.cast(tf.sign(y_true[:, 2]), 'bool')), 'float')
            s2 = tf.cast(tf.logical_and(tf.greater(y_hat[:, 5], confidance), tf.cast(tf.sign(y_true[:, 5]), 'bool')), 'float')
            s3 = tf.cast(tf.logical_and(tf.greater(y_hat[:, 8], confidance), tf.cast(tf.sign(y_true[:, 8]), 'bool')), 'float')
            s4 = tf.cast(tf.logical_and(tf.greater(y_hat[:, 11], confidance), tf.cast(tf.sign(y_true[:, 11]), 'bool')), 'float')
            s5 = tf.cast(tf.logical_and(tf.greater(y_hat[:, 14], confidance), tf.cast(tf.sign(y_true[:, 14]), 'bool')), 'float')
            s6 = tf.cast(tf.logical_and(tf.greater(y_hat[:, 17], confidance), tf.cast(tf.sign(y_true[:, 17]), 'bool')), 'float')
            s7 = tf.cast(tf.logical_and(tf.greater(y_hat[:, 20], confidance), tf.cast(tf.sign(y_true[:, 20]), 'bool')), 'float')

    # count_keypoints = tf.cast(tf.count_nonzero(y_true[:,2]),'float')+tf.cast(tf.count_nonzero(y_true[:,5]),'float')+\
    #                   tf.cast(tf.count_nonzero(y_true[:,8]),'float')+tf.cast(tf.count_nonzero(y_true[:,11]),'float')+\
    #                   tf.cast(tf.count_nonzero(y_true[:,14]),'float')+tf.cast(tf.count_nonzero(y_true[:,17]),'float')+\
    #                   tf.cast(tf.count_nonzero(y_true[:,20]),'float')

            wpar = [s1, s2, s3, s4, s5, s6, s7]
        return wpar


    def auc_us_single(self,wpar):
        return tf.metrics.auc(wpar[0], wpar[1])[1]

    def auc_us_parallel(self,y_true, y_pred):
        with tf.variable_scope('auc_us_parallel'):
            y_pred = tf.reshape(y_pred, [-1, 26])
            y_true = tf.reshape(y_true, [-1, 26])
    # auc_st = tf.metrics.auc(y_true[:,21], y_pred[:,21])[1]
            wpar = (y_true[:, 21], y_pred[:, 21])
        return wpar


    def AUC_all_single (self,wpar):
        with tf.variable_scope('AUC_all_single'):

            y_true = wpar[0]
            y_pred = wpar[1]

    # print(y_true.get_shape())
    # print(y_pred.get_shape())
            auc_st1 = tf.metrics.auc(y_true[:, 2], y_pred[:, 2])
            auc_st2 = tf.metrics.auc(y_true[:, 5], y_pred[:, 5])
            auc_st3 = tf.metrics.auc(y_true[:, 8], y_pred[:, 8])
            auc_st4 = tf.metrics.auc(y_true[:, 11], y_pred[:, 11])
            auc_st5 = tf.metrics.auc(y_true[:, 14], y_pred[:, 14])
            auc_st6 = tf.metrics.auc(y_true[:, 17], y_pred[:, 17])
            auc_st7 = tf.metrics.auc(y_true[:, 20], y_pred[:, 20])

            auc_st = (auc_st1[1]+auc_st2[1]+auc_st3[1]+auc_st4[1]+auc_st5[1]+auc_st6[1]+auc_st7[1])/7
        return auc_st

    def AUC_all_parallel (self,y_true, y_pred):
        with tf.variable_scope('AUC_all_parallel'):
            y_pred = tf.reshape(y_pred, [-1, 26])
            y_true = tf.keras.backend.sign(tf.reshape(y_true, [-1, 26]))


            wpar = [y_true, y_pred]
        return wpar

    def body (self, y_hat_re,y_true_re, conf, increase):


        tp1 = tf.cast(tf.logical_and(tf.greater(y_hat_re[:, 0], conf), tf.cast(tf.sign(y_true_re[:, 2]), 'bool')),
                      'float')
        tp2 = tf.cast(tf.logical_and(tf.greater(y_hat_re[:, 1], conf), tf.cast(tf.sign(y_true_re[:, 5]), 'bool')),
                      'float')
        tp3 = tf.cast(tf.logical_and(tf.greater(y_hat_re[:, 2], conf), tf.cast(tf.sign(y_true_re[:, 8]), 'bool')),
                      'float')
        tp4 = tf.cast(tf.logical_and(tf.greater(y_hat_re[:, 3], conf), tf.cast(tf.sign(y_true_re[:, 11]), 'bool')),
                      'float')
        tp5 = tf.cast(tf.logical_and(tf.greater(y_hat_re[:, 4], conf), tf.cast(tf.sign(y_true_re[:, 14]), 'bool')),
                      'float')
        tp6 = tf.cast(tf.logical_and(tf.greater(y_hat_re[:, 5], conf), tf.cast(tf.sign(y_true_re[:, 17]), 'bool')),
                      'float')
        tp7 = tf.cast(tf.logical_and(tf.greater(y_hat_re[:, 6], conf), tf.cast(tf.sign(y_true_re[:, 20]), 'bool')),
                      'float')
        tp8 = tf.cast(tf.logical_and(tf.greater(y_hat_re[:, 7], conf), tf.cast(tf.sign(y_true_re[:, 21]), 'bool')),
                      'float')

        count_keypoints_pos_results = tf.reduce_sum(tf.cast(tf.greater(y_hat_re[:, 0], conf), 'float'))+ \
                                      tf.reduce_sum(tf.cast(tf.greater(y_hat_re[:, 1], conf), 'float'))+ \
                                      tf.reduce_sum(tf.cast(tf.greater(y_hat_re[:, 2], conf), 'float'))+ \
                                      tf.reduce_sum(tf.cast(tf.greater(y_hat_re[:, 3], conf), 'float'))+ \
                                      tf.reduce_sum(tf.cast(tf.greater(y_hat_re[:, 4], conf), 'float'))+ \
                                      tf.reduce_sum(tf.cast(tf.greater(y_hat_re[:, 5], conf), 'float'))+ \
                                      tf.reduce_sum(tf.cast(tf.greater(y_hat_re[:, 6], conf), 'float'))+ \
                                      tf.reduce_sum(tf.cast(tf.greater(y_hat_re[:, 7], conf), 'float'))

        count_keypoints_totalcases = tf.cast(tf.count_nonzero(y_true_re[:, 2]), 'float')+ \
                                     tf.cast(tf.count_nonzero(y_true_re[:, 5]), 'float')+ \
                                     tf.cast(tf.count_nonzero(y_true_re[:, 8]), 'float')+ \
                                     tf.cast(tf.count_nonzero(y_true_re[:, 11]), 'float')+ \
                                     tf.cast(tf.count_nonzero(y_true_re[:, 14]), 'float')+ \
                                     tf.cast(tf.count_nonzero(y_true_re[:, 17]), 'float')+ \
                                     tf.cast(tf.count_nonzero(y_true_re[:, 20]), 'float')+ \
                                     tf.cast(tf.count_nonzero(y_true_re[:, 21]), 'float')

        summ = tf.reduce_sum(tp1) + tf.reduce_sum(tp2) + tf.reduce_sum(tp3) + tf.reduce_sum(tp4) + \
               tf.reduce_sum(tp5) + tf.reduce_sum(tp6) + tf.reduce_sum(tp7)+ tf.reduce_sum(tp8)
        RECAL = tf.where(tf.logical_not(tf.equal(count_keypoints_totalcases, 0)), x=(summ / count_keypoints_totalcases),y=-1)
        PERCISION = tf.where(tf.logical_not(tf.equal(count_keypoints_pos_results, 0)),x=(summ / count_keypoints_pos_results), y=-1)



        return [tp1, tp2, tp3, tp4, tp5, tp6, tp7, tf.add(conf, increase)]

    def condition(self, y_hat_re,y_true_re, conf, increase):
        return tf.less(conf, 1)

    def new_acuracy_single (self,wpar):
        with tf.variable_scope('new_acuracy_single'):
            list = [2, 5, 8, 11, 14, 17, 20, 21]
            count_keypoints_totalcases = 0
            count_keypoints_pos_results = 0
            conf = 0.5
            for i in range (0,8):
                count_keypoints_pos_results += tf.reduce_sum(tf.cast(tf.greater(wpar[0][:, i], conf), 'float'))
            for i in list:
                # count_keypoints_pos_results += tf.reduce_sum(tf.cast(tf.greater(wpar[0][:, i], conf), 'float'))
                count_keypoints_totalcases += tf.cast(tf.count_nonzero(wpar[1][:, i]), 'float')

            summ = tf.reduce_sum(wpar[2]) + tf.reduce_sum(wpar[3]) + tf.reduce_sum(wpar[4]) + tf.reduce_sum(wpar[5]) + \
                   tf.reduce_sum(wpar[6]) + tf.reduce_sum(wpar[7]) + tf.reduce_sum(wpar[8]) + tf.reduce_sum(wpar[9])
            RECAL = tf.where(tf.logical_not(tf.equal(count_keypoints_totalcases, 0)), x=(summ / count_keypoints_totalcases), y=-1)
            PERCISION = tf.where(tf.logical_not(tf.equal(count_keypoints_pos_results, 0)), x=(summ / count_keypoints_pos_results), y=-1)
            FP_rate = tf.where(tf.logical_not(tf.equal(count_keypoints_totalcases, 0)), x=((count_keypoints_pos_results - summ)/count_keypoints_pos_results), y=-1)

            mAP1 = tf.metrics.average_precision_at_k(tf.cast(tf.sign(wpar[1][:, 2]),tf.int64), wpar[0][:,0], 1)[0]
            mAP2 = tf.metrics.average_precision_at_k(tf.cast(tf.sign(wpar[1][:, 5]),tf.int64), wpar[0][:, 1], 1)[0]
            mAP3 = tf.metrics.average_precision_at_k(tf.cast(tf.sign(wpar[1][:, 8]),tf.int64), wpar[0][:, 2], 1)[0]
            mAP4 = tf.metrics.average_precision_at_k(tf.cast(tf.sign(wpar[1][:, 11]),tf.int64), wpar[0][:, 3], 1)[0]
            mAP5 = tf.metrics.average_precision_at_k(tf.cast(tf.sign(wpar[1][:, 14]),tf.int64), wpar[0][:, 4], 1)[0]
            mAP6 = tf.metrics.average_precision_at_k(tf.cast(tf.sign(wpar[1][:, 17]),tf.int64), wpar[0][:, 5], 1)[0]
            mAP7 = tf.metrics.average_precision_at_k(tf.cast(tf.sign(wpar[1][:, 20]),tf.int64), wpar[0][:, 6], 1)[0]
            mAP8 = tf.metrics.average_precision_at_k(tf.cast(tf.sign(wpar[1][:, 21]),tf.int64), wpar[0][:, 7], 1)[0]
            mAP = tf.add_n([mAP1+mAP2+mAP3+mAP4+mAP5+mAP6+mAP7+mAP8])/8.0

        return RECAL, PERCISION,mAP, FP_rate, count_keypoints_totalcases, count_keypoints_pos_results, summ

    def new_acuracy_parallel(self,y_true, y_pred):
        with tf.variable_scope('new_acuracy_parallel'):

            increase = 0.5
            conf = 0.5
            y_hat_re = tf.reshape(y_pred, [-1, 8])
            y_true_re = tf.reshape(y_true, [-1, 26])
            # wpar = tf.while_loop(self.condition, self.body, [y_hat_re, y_true_re, conf, increase])

            tp1 = tf.cast(tf.logical_and(tf.greater(y_hat_re[:, 0], conf), tf.cast(tf.sign(y_true_re[:, 2]), 'bool')),
                          'float')
            tp2 = tf.cast(tf.logical_and(tf.greater(y_hat_re[:, 1], conf), tf.cast(tf.sign(y_true_re[:, 5]), 'bool')),
                          'float')
            tp3 = tf.cast(tf.logical_and(tf.greater(y_hat_re[:, 2], conf), tf.cast(tf.sign(y_true_re[:, 8]), 'bool')),
                          'float')
            tp4 = tf.cast(tf.logical_and(tf.greater(y_hat_re[:, 3], conf), tf.cast(tf.sign(y_true_re[:, 11]), 'bool')),
                          'float')
            tp5 = tf.cast(tf.logical_and(tf.greater(y_hat_re[:, 4], conf), tf.cast(tf.sign(y_true_re[:, 14]), 'bool')),
                          'float')
            tp6 = tf.cast(tf.logical_and(tf.greater(y_hat_re[:, 5], conf), tf.cast(tf.sign(y_true_re[:, 17]), 'bool')),
                          'float')
            tp7 = tf.cast(tf.logical_and(tf.greater(y_hat_re[:, 6], conf), tf.cast(tf.sign(y_true_re[:, 20]), 'bool')),
                          'float')
            tp8 = tf.cast(tf.logical_and(tf.greater(y_hat_re[:, 7], conf), tf.cast(tf.sign(y_true_re[:, 21]), 'bool')),
                          'float')


            wpar = (y_hat_re, y_true_re, tp1, tp2, tp3, tp4, tp5, tp6, tp7, tp8)

        return wpar









