from __future__ import division
import tensorflow as tf
import numpy as np

'''
These functions are the metrics that are needed for performance monitoring. The functions are known with their names and 
tailored to the spesific dataset made to train Human Gestures.
'''

def mAP_RealImage (Y_true, Y_pred, confidance = 0.2, iou_score=0.5, iou_same_person_score = 0.5):

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

def Distance (y_true, y_pred):

    y_hat = tf.reshape(y_pred, [-1, 26])
    y = tf.cast(tf.reshape(y_true, [-1, 26]), tf.float32)

    c1 = tf.reduce_sum(tf.sqrt(tf.sign(y[:, 2]) *
                               (tf.keras.backend.square(y[:, 0] - y_hat[:, 0]) + tf.keras.backend.square(y[:, 1] - y_hat[:, 1]))))
    count1 = tf.cast(tf.count_nonzero(tf.sign(y[:, 2])), 'float')
    # mean1 = tf.where(tf.equal(count1, 0.0), x = (c1 / count1), y = 0)

    c2 = tf.reduce_sum(tf.sqrt(tf.sign(y[:, 5]) *
                               (tf.keras.backend.square(y[:, 3] - y_hat[:, 3]) + tf.keras.backend.square(y[:, 4] - y_hat[:, 4]))))
    count2 = tf.cast(tf.count_nonzero(y[:, 5]), 'float')
    # mean2 = tf.where(tf.equal(count2, 0.0), x = (c2 / count2), y = 0)

    c3 = tf.reduce_sum(tf.sqrt(tf.sign(y[:, 8]) *
                               (tf.keras.backend.square(y[:, 6] - y_hat[:, 6]) + tf.keras.backend.square(y[:, 7] - y_hat[:, 7]))))
    count3 = tf.cast(tf.count_nonzero(tf.sign(y[:, 8])), 'float')
    # mean3 = tf.where(tf.equal(count3, 0.0), x = (c3 / count3), y = 0)

    c4 = tf.reduce_sum(tf.sqrt(tf.sign(y[:, 11]) *
                               (tf.keras.backend.square(y[:, 9] - y_hat[:, 9]) + tf.keras.backend.square(y[:, 10] - y_hat[:, 10]))))
    count4 = tf.cast(tf.count_nonzero(tf.sign(y[:, 11])), 'float')
    # mean4 = tf.where(tf.equal(count4, 0.0), x = (c4 / count4), y = 0)

    c5 = tf.reduce_sum(tf.sqrt(tf.sign(y[:, 14]) *
                               (tf.keras.backend.square(y[:, 12] - y_hat[:, 12]) + tf.keras.backend.square(y[:, 13] - y_hat[:, 13]))))
    count5 = tf.cast(tf.count_nonzero(tf.sign(y[:, 14])), 'float')
    # mean5 = tf.where(tf.equal(count5, 0.0), x = (c5 / count5), y = 0)

    c6 = tf.reduce_sum(tf.sqrt(tf.sign(y[:, 17]) *
                               (tf.keras.backend.square(y[:, 15] - y_hat[:, 15]) + tf.keras.backend.square(y[:, 16] - y_hat[:, 16]))))
    count6 = tf.cast(tf.count_nonzero(tf.sign(y[:, 17])), 'float')
    # mean6 = tf.where(tf.equal(count6, 0.0), x = (c6 / count6), y = 0)

    c7 = tf.reduce_sum(tf.sqrt(tf.sign(y[:, 20]) *
                               (tf.keras.backend.square(y[:, 18] - y_hat[:, 18]) + tf.keras.backend.square(y[:, 19] - y_hat[:, 19]))))
    count7 = tf.cast(tf.count_nonzero(tf.sign(y[:, 20])), 'float')
    # mean7 = tf.where(tf.equal(count7, 0.0), x = (c7 / count7), y = 0)
    summ = c1+c2+c3+c4+c5+c6+c7
    count = count1+count2+count3+count4+count5+count6+count7

    mean_distance = tf.where(tf.logical_not(tf.equal(count, 0)), x= (summ / tf.cast(count, 'float')), y=0)

    count8 = tf.cast(tf.count_nonzero(y[:, 21]), 'float')

    # counter = tf.sign(mean1)+tf.sign(mean2)+tf.sign(mean3)+tf.sign(mean4)+tf.sign(mean5)+tf.sign(mean6)+tf.sign(mean7)

    # mean_distance = (mean1+mean2+mean3+mean4+mean5+mean6+mean7) / counter

    xA = tf.maximum((y[:, 22] - y[:,24] /2), (y_hat[:, 22] - y_hat[:,24] /2))
    yA = tf.maximum(y[:, 23] - y[:,25] /2, y_hat[:, 23] - y_hat[:,25] /2)
    xB = tf.minimum(y[:, 22] + y[:,24] /2, y_hat[:, 22] + y_hat[:,24] /2)
    yB = tf.minimum(y[:, 23] + y[:,25] /2, y_hat[:, 23] + y_hat[:,25] /2)
    interArea = tf.sign(y[:, 21]) * (tf.maximum(0.0, xB - xA) * tf.maximum(0.0, yB - yA))
    boxAArea = tf.sign(y[:, 21]) * ((y_hat[:, 24]) * (y_hat[:, 25]))
    boxBArea = tf.sign(y[:, 21]) * ((y[:, 24]) * (y[:, 25]))
    iou = tf.abs(interArea / (boxAArea + boxBArea - interArea))
    kk = tf.reduce_sum(tf.cast(tf.greater(iou, 0.0), 'float'))
    mAP_at_iou_half = (kk)/count8
    # mean_distance = sum
    return (mean_distance, mAP_at_iou_half)

def map_k(y_true, y_pred,k=1):          # k shows the number of guesses that are ranked correct, 1 and 5 are usual but here its either true or not
    y_pred = tf.reshape(y_pred, [-1, 26])
    y_true = tf.cast(tf.reshape(y_true, [-1, 26]),tf.int64)

    at1=tf.metrics.average_precision_at_k(y_true[:,2],y_pred[:,2],k)
    at2= tf.metrics.average_precision_at_k(y_true[:, 5], y_pred[:, 5], k)
    at3= tf.metrics.average_precision_at_k(y_true[:, 8], y_pred[:, 8], k)
    at4 = tf.metrics.average_precision_at_k(y_true[:, 11], y_pred[:, 11], k)
    at5 = tf.metrics.average_precision_at_k(y_true[:, 14], y_pred[:, 14], k)
    at6 = tf.metrics.average_precision_at_k(y_true[:, 17], y_pred[:, 17], k)
    at7 = tf.metrics.average_precision_at_k(y_true[:, 20], y_pred[:, 20], k)
    at8 = tf.metrics.average_precision_at_k(y_true[:, 21], y_pred[:, 21], k)

    at=(at1[0]+at2[0]+at3[0]+at4[0]+at5[0]+at6[0]+at7[0]+at8[0])/8

    # auc_st = tf.metrics.auc(y_true[:,21], y_pred[:,21])[1]
    tf.keras.backend.get_session().run(tf.local_variables_initializer())
    au = tf.keras.backend.get_session().run(at)
    return au

def auc_us(y_true, y_pred):
    y_pred = tf.reshape(y_pred, [-1, 26])
    y_true = tf.reshape(y_true, [-1, 26])

    auc_st = tf.metrics.auc(y_true[:,21], y_pred[:,21])[1]
    return auc_st

def read_and_decode(imagesize, filename_queue, batch_size, num_threads):

    reader = tf.TFRecordReader()

    _, serialized_example = reader.read(filename_queue)

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

    # resized_annotation = tf.image.resize_image_with_crop_or_pad(image=annotation, target_height=IMAGE_HEIGHT,target_width=IMAGE_WIDTH)


    images, annotations = tf.train.shuffle_batch( [resized_image, annotation],
                                                 batch_size= batch_size,
                                                 capacity=5000,
                                                 num_threads= num_threads,
                                                 min_after_dequeue=1000)

    return images, annotations

def AP (y_hat, y_true):

    # y_hat = np.reshape(y_hat, [-1, 26])
    # y_true = np.reshape(y_true, [-1, 26])

    TP = 0
    FP = 0
    FN = 0

    TP += tf.reduce_sum(tf.cast(tf.logical_and(y_hat[:,2], y_true[:,2]),'float'))
    TP += tf.reduce_sum(tf.cast(tf.logical_and(y_hat[:,5], y_true[:,5]),'float'))
    TP += tf.reduce_sum(tf.cast(tf.logical_and(y_hat[:,8], y_true[:,8]),'float'))
    TP += tf.reduce_sum(tf.cast(tf.logical_and(y_hat[:,11], y_true[:,11]),'float'))
    TP += tf.reduce_sum(tf.cast(tf.logical_and(y_hat[:,14], y_true[:,14]),'float'))
    TP += tf.reduce_sum(tf.cast(tf.logical_and(y_hat[:,17], y_true[:,17]),'float'))
    TP += tf.reduce_sum(tf.cast(tf.logical_and(y_hat[:,20], y_true[:,20]),'float'))
    TP += tf.reduce_sum(tf.cast(tf.logical_and(y_hat[:, 21], y_true[:, 21]),'float'))

    FP += tf.reduce_sum(tf.cast(tf.logical_and(y_hat[:,2], tf.logical_not(y_true[:,2])),'float'))
    FP += tf.reduce_sum(tf.cast(tf.logical_and(y_hat[:,5], tf.logical_not(y_true[:,5])),'float'))
    FP += tf.reduce_sum(tf.cast(tf.logical_and(y_hat[:,8], tf.logical_not(y_true[:,8])),'float'))
    FP += tf.reduce_sum(tf.cast(tf.logical_and(y_hat[:,11], tf.logical_not(y_true[:,11])),'float'))
    FP += tf.reduce_sum(tf.cast(tf.logical_and(y_hat[:,14], tf.logical_not(y_true[:,14])),'float'))
    FP += tf.reduce_sum(tf.cast(tf.logical_and(y_hat[:,17], tf.logical_not(y_true[:,17])),'float'))
    FP += tf.reduce_sum(tf.cast(tf.logical_and(y_hat[:,20], tf.logical_not(y_true[:,20])),'float'))
    FP += tf.reduce_sum(tf.cast(tf.logical_and(y_hat[:,21], tf.logical_not(y_true[:,21])),'float'))

    FN += tf.reduce_sum(tf.cast(tf.logical_and(tf.logical_not(y_hat[:,2]), y_true[:,2]),'float'))
    FN += tf.reduce_sum(tf.cast(tf.logical_and(tf.logical_not(y_hat[:,5]), y_true[:,5]),'float'))
    FN += tf.reduce_sum(tf.cast(tf.logical_and(tf.logical_not(y_hat[:,8]), y_true[:,8]),'float'))
    FN += tf.reduce_sum(tf.cast(tf.logical_and(tf.logical_not(y_hat[:,11]), y_true[:,11]),'float'))
    FN += tf.reduce_sum(tf.cast(tf.logical_and(tf.logical_not(y_hat[:,14]), y_true[:,14]),'float'))
    FN += tf.reduce_sum(tf.cast(tf.logical_and(tf.logical_not(y_hat[:,17]), y_true[:,17]),'float'))
    FN += tf.reduce_sum(tf.cast(tf.logical_and(tf.logical_not(y_hat[:,20]), y_true[:,20]),'float'))
    FN += tf.reduce_sum(tf.cast(tf.logical_and(tf.logical_not(y_hat[:,21]), y_true[:,21]),'float'))

    percision = (TP) / (TP + FP)
    recall = (TP) / (TP + FN)   #as the treshold gets bigger the recall gets smaller



    # predictions = [y_hat[:,2], y_hat[:,5], y_hat[:,8], y_hat[:,11], y_hat[:,14], y_hat[:,17], y_hat[:,20], y_hat[:,21]]
    # labels = [y_true[:,2], y_true[:,5], y_true[:,8], y_true[:,11], y_true[:,14], y_true[:,17], y_true[:,20], y_true[:,21]]
    # y_hat = tf.reshape(y_hat, [-1, 26])
    # predictions = y_hat [:,2]
    # y_true = tf.reshape(y_true, [-1, 26])
    # labels = tf.cast(y_true [:, 2], tf.int64)
    #
    # map = tf.metrics.average_precision_at_k(y_true[:,2],tf.cast(y_hat[:,2], tf.float64),k = 1)

    return (percision, recall)

def Accu (y_hat, y_true, confidance = 0.5):

    y_hat = tf.reshape(y_hat, [-1, 26])
    y_true = tf.reshape(y_true, [-1, 26])

    # true1 = tf.cast(tf.equal(tf.to_int32(tf.round(y_hat[:, 2])), tf.to_int32(tf.round(y_true[:, 2]))), 'float')
    # true2 = tf.cast(tf.equal(tf.to_int32(tf.round(y_hat[:, 5])), tf.to_int32(tf.round(y_true[:, 5]))),'float')
    # true3 = tf.cast(tf.equal(tf.to_int32(tf.round(y_hat[:, 8])), tf.to_int32(tf.round(y_true[:, 8]))),'float')
    # true4 = tf.cast(tf.equal(tf.to_int32(tf.round(y_hat[:, 11])), tf.to_int32(tf.round(y_true[:, 11]))),'float')
    # true5 = tf.cast(tf.equal(tf.to_int32(tf.round(y_hat[:, 14])), tf.to_int32(tf.round(y_true[:, 14]))),'float')
    # true6 = tf.cast(tf.equal(tf.to_int32(tf.round(y_hat[:, 17])), tf.to_int32(tf.round(y_true[:, 17]))),'float')
    # true7 = tf.cast(tf.equal(tf.to_int32(tf.round(y_hat[:, 20])), tf.to_int32(tf.round(y_true[:, 20]))),'float')
    # true8 = tf.cast(tf.equal(tf.to_int32(tf.round(y_hat[:, 21])), tf.to_int32(tf.round(y_true[:, 21]))), 'float')

    # summ = tf.metrics.true_positives(y_hat[:, 2], y_true[:, 2])[1]+tf.metrics.true_positives(y_hat[:, 5], y_true[:, 5])[1]+tf.metrics.true_positives(y_hat[:, 8], y_true[:, 8])[1]\
    # +tf.metrics.true_positives(y_hat[:, 11], y_true[:, 11])[1]+tf.metrics.true_positives(y_hat[:, 14], y_true[:, 14])[1]\
    # +tf.metrics.true_positives(y_hat[:, 17], y_true[:, 17])[1]+tf.metrics.true_positives(y_hat[:, 17], y_true[:, 17])[1]\
    # +tf.metrics.true_positives(y_hat[:, 20], y_true[:, 20])[1]

    summ = tf.reduce_sum(tf.cast(tf.logical_and(tf.greater(y_hat[:, 2],confidance), tf.cast(tf.sign(y_true[:, 2]),'bool')),'float'))+\
        tf.reduce_sum(tf.cast(tf.logical_and(tf.greater(y_hat[:, 5],confidance), tf.cast(tf.sign(y_true[:, 5]), 'bool')),'float'))+ \
        tf.reduce_sum(tf.cast(tf.logical_and(tf.greater(y_hat[:, 8],confidance), tf.cast(tf.sign(y_true[:, 8]), 'bool')),'float'))+ \
        tf.reduce_sum(tf.cast(tf.logical_and(tf.greater(y_hat[:, 11],confidance), tf.cast(tf.sign(y_true[:, 11]), 'bool')),'float'))+ \
        tf.reduce_sum(tf.cast(tf.logical_and(tf.greater(y_hat[:, 14],confidance), tf.cast(tf.sign(y_true[:, 14]), 'bool')),'float'))+ \
        tf.reduce_sum(tf.cast(tf.logical_and(tf.greater(y_hat[:, 17],confidance), tf.cast(tf.sign(y_true[:, 17]), 'bool')),'float')) + \
        tf.reduce_sum(tf.cast(tf.logical_and(tf.greater(y_hat[:, 20],confidance), tf.cast(tf.sign(y_true[:, 20]), 'bool')),'float'))


    count_keypoints = tf.cast(tf.count_nonzero(y_true[:,2]),'float')+tf.cast(tf.count_nonzero(y_true[:,5]),'float')+\
                      tf.cast(tf.count_nonzero(y_true[:,8]),'float')+tf.cast(tf.count_nonzero(y_true[:,11]),'float')+\
                      tf.cast(tf.count_nonzero(y_true[:,14]),'float')+tf.cast(tf.count_nonzero(y_true[:,17]),'float')+\
                      tf.cast(tf.count_nonzero(y_true[:,20]),'float')

    ac = tf.where(tf.logical_not(tf.equal(count_keypoints, 0)), x=(summ / count_keypoints), y=0)
    # count = 8
    # ac = (tf.reduce_mean(true1) + tf.reduce_mean(true2) + tf.reduce_mean(true3) + tf.reduce_mean(true8) + \
    #        tf.reduce_mean(true4) + tf.reduce_mean(true5) + tf.reduce_mean(true6) + tf.reduce_mean(true7))
    # ac = mean_distance/int(count)

    # me1 = tf.count_nonzero(tf.cast(tf.equal(tf.to_int32(tf.round(y_hat[:, 2])), tf.to_int32(tf.round(y_true[:, 2]))), 'float'))
    # m1 = tf.count_nonzero(tf.to_int32(tf.round(y_true[:, 2])))
    #
    # me2 = tf.count_nonzero(tf.to_int32(tf.round(y_hat[:, 5])))
    # m2 = tf.count_nonzero(tf.to_int32(tf.round(y_true[:, 5])))
    #
    # me3 = tf.count_nonzero(tf.to_int32(tf.round(y_hat[:, 8])))
    # m3 = tf.count_nonzero(tf.to_int32(tf.round(y_true[:, 8])))
    #
    # me4 = tf.count_nonzero(tf.to_int32(tf.round(y_hat[:, 11])))
    # m4 = tf.count_nonzero(tf.to_int32(tf.round(y_true[:, 11])))
    #
    # me5 = tf.count_nonzero(tf.to_int32(tf.round(y_hat[:, 14])))
    # m5 = tf.count_nonzero(tf.to_int32(tf.round(y_true[:, 14])))
    #
    # me6 = tf.count_nonzero(tf.to_int32(tf.round(y_hat[:, 17])))
    # m6 = tf.count_nonzero(tf.to_int32(tf.round(y_true[:, 17])))
    #
    # me7 = tf.count_nonzero(tf.to_int32(tf.round(y_hat[:, 20])))
    # m7 = tf.count_nonzero(tf.to_int32(tf.round(y_true[:, 20])))
    #
    # me8 = tf.count_nonzero(tf.to_int32(tf.round(y_hat[:, 21])))
    # m8 = tf.count_nonzero(tf.to_int32(tf.round(y_true[:, 21])))
    #
    # me = me1+me2+me3+me4+me5+me6+me7+me8
    # m = m1+m2+m3+m4+m5+m6+m7+m8
    # tf.where(tf.logical_not(tf.equal(m, 0)), x=(me / tf.cast(count, 'float')), y=0)

    return ac


def AUC_all (y_true, y_pred):


    y_pred = tf.reshape(y_pred, [-1, 26])
    y_true = tf.keras.backend.sign(tf.reshape(y_true, [-1, 26]))

    auc_st1 = tf.metrics.auc(y_true[:, 2], y_pred[:, 2])[1]
    auc_st2 = tf.metrics.auc(y_true[:, 5], y_pred[:, 5])[1]
    auc_st3 = tf.metrics.auc(y_true[:, 8], y_pred[:, 8])[1]
    auc_st4 = tf.metrics.auc(y_true[:, 11], y_pred[:, 11])[1]
    auc_st5 = tf.metrics.auc(y_true[:, 14], y_pred[:, 14])[1]
    auc_st6 = tf.metrics.auc(y_true[:, 17], y_pred[:, 17])[1]
    auc_st7 = tf.metrics.auc(y_true[:, 20], y_pred[:, 20])[1]

    auc_st = (auc_st1+auc_st2+auc_st3+auc_st4+auc_st5+auc_st6+auc_st7)/7
    # tf.keras.backend.get_session().run(tf.local_variables_initializer())
    # au = tf.keras.backend.get_session().run(auc_st)
    return auc_st




def miou(y_hat, y_true):
    per = 0
    y_hat = tf.reshape(y_hat, [-1, 26])
    y_true = tf.reshape(y_true, [-1, 26])
    for k in range (0,11):

        y_hat_tmp = tf.greater(y_hat, k/10)
        y_true_tmp = tf.greater(y_true, k/10)

        percision, recall = AP(y_hat_tmp, y_true_tmp)
        # print recall
        percision = tf.where(tf.is_nan(percision), 0., percision)   # as the treshold gets bigger, the recall gets smaller
        per+=percision

    return (per/10)
