from __future__ import division
import tensorflow as tf
import numpy as np
from sklearn.metrics import average_precision_score



'''
These functions are the metrics that are needed for performance monitoring. The functions are known with their names and 
tailored to the spesific dataset made to train Human Gestures.
'''


def _parse_function(proto):
    # define your tfrecord again. Remember that you saved your image as a string.
    keys_to_features = {'height': tf.io.FixedLenFeature([], tf.int64),
        'width': tf.FixedLenFeature([], tf.int64),
        'image_raw': tf.FixedLenFeature([], tf.string),
        'mask_raw': tf.FixedLenFeature([], tf.string)}

    # Load one example
    parsed_features = tf.io.parse_single_example(proto, keys_to_features)

    # Turn your saved image string into an array
    parsed_features['image_raw'] = tf.decode_raw(parsed_features['image_raw'], tf.uint8)
    parsed_features['mask_raw'] = tf.decode_raw(parsed_features['mask_raw'], tf.float16)

    return parsed_features['image_raw'], parsed_features["mask_raw"]

def preprocessing (input):
    # beta = tf.Variable(tf.constant(0.0, shape=[n_out]),
    #                    name='beta', trainable=True)
    # gamma = tf.Variable(tf.constant(1.0, shape=[n_out]),
    #                     name='gamma', trainable=True)
    # batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], name='moments')
    # ema = tf.train.ExponentialMovingAverage(decay=0.5)
    #
    # def mean_var_with_update():
    #     ema_apply_op = ema.apply([batch_mean, batch_var])
    #     with tf.control_dependencies([ema_apply_op]):
    #         return tf.identity(batch_mean), tf.identity(batch_var)
    #
    # mean, var = tf.cond(phase_train,
    #                     mean_var_with_update,
    #                     lambda: (ema.average(batch_mean), ema.average(batch_var)))
    # normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
    mean, variance = tf.nn.moments(input,axes=[0,1,2],keep_dims=False)
    output = tf.nn.batch_normalization(input,mean,variance,offset=0,scale=1,variance_epsilon=1e-7)


    return mean, variance, output
    # return mean,variance

def read_image_tf_data(imagesize, filename_queue, batch_size, num_threads):


    # This works with arrays as well
    dataset = tf.data.TFRecordDataset(filename_queue)

    # Maps the parser on every filepath in the array. You can set the number of parallel loaders here
    dataset = dataset.map(_parse_function, num_parallel_calls=num_threads)

    # This dataset will go on forever
    dataset = dataset.repeat()

    # Set the number of datapoints you want to load and shuffle
    dataset = dataset.shuffle(500)

    # Set the batchsize
    dataset = dataset.batch(batch_size)

    # Create an iterator
    iterator = dataset.make_one_shot_iterator()

    # Create your tf representation of the iterator
    image, label = iterator.get_next()

    # annotation_shape = tf.stack([10, 10, 26])
    label = tf.reshape(label, [-1, 10, 10, 26])

    # Bring your picture back in shape
    image = tf.reshape(image, [-1, imagesize, imagesize, 3])

    # Create a one hot array for your labels
    # label = tf.one_hot(label, NUM_CLASSES)
    image = image / 255
    # image1 = image[:,:,:,0] /
    # image2 = image[:, :, :, 1] /
    # image3 = image[:, :, :, 2] /
    # image = tf.concat([image1,image2,image3], 3)
    # print (image.type)

    # image = tf.image.per_image_standardization(image)
    mean, std, image = preprocessing(image)
    # shap = image.shape
    # mean1 = tf.concat([tf.reduce_sum(image[:,:,:,0])/shap[0],
    #                    tf.reduce_sum(image[:,:,:,1])/shap[1],
    #                    tf.reduce_sum(image[:,:,:,2])/shap[2]], 0)

    return image, label
    # return image, label


def read_and_decode(imagesize, filename_queue, batch_size, num_threads):

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(
      serialized_example,
      features={
        'height': tf.FixedLenFeature([], tf.int64),
        'width': tf.FixedLenFeature([], tf.int64),
        'image_raw': tf.FixedLenFeature([], tf.string),
        'mask_raw': tf.FixedLenFeature([], tf.string)
        })

    image = tf.decode_raw(features['image_raw'], tf.uint8)
    annotation = tf.decode_raw(features['mask_raw'], tf.float16)

    height = tf.cast(features['height'], tf.int32)
    width = tf.cast(features['width'], tf.int32)

    image_shape = tf.stack([height, width, 3])
    annotation_shape = tf.stack([10, 10, 26])
    annotation = tf.reshape(annotation, annotation_shape)

    image = tf.reshape(image, image_shape)

    image = image /255
    resized_image = tf.image.resize_image_with_crop_or_pad(image=image, target_height=imagesize, target_width=imagesize)    #redundant in case of change later

    images, annotations = tf.train.batch( [resized_image, annotation],
                                                 batch_size= batch_size,
                                                 enqueue_many=False,
                                                 capacity=500,
                                                 num_threads= num_threads)

    return images, annotations


class Out_Metric(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self._data = []

    def on_epoch_end(self, batch, logs={}):
        X_val, y_val = self.validation_data[0], self.validation_data[1]
        y_predict = np.asarray(self.model.predict(X_val))

        y_val = np.argmax(y_val, axis=1)
        y_predict = np.argmax(y_predict, axis=1)
        y_hat_val = np.reshape(y_predict, [-1, 26])
        y_true_val = np.reshape(y_val, [-1, 26])
        sk_map1 = average_precision_score(np.sign(y_true_val[:, 2]).astype(int), y_hat_val[:, 2])
        sk_map2 = average_precision_score(np.sign(y_true_val[:, 5]).astype(int), y_hat_val[:, 5])
        sk_map3 = average_precision_score(np.sign(y_true_val[:, 8]).astype(int), y_hat_val[:, 8])
        sk_map4 = average_precision_score(np.sign(y_true_val[:, 11]).astype(int), y_hat_val[:, 11])
        sk_map5 = average_precision_score(np.sign(y_true_val[:, 14]).astype(int), y_hat_val[:, 14])
        sk_map6 = average_precision_score(np.sign(y_true_val[:, 17]).astype(int), y_hat_val[:, 17])
        sk_map7 = average_precision_score(np.sign(y_true_val[:, 20]).astype(int), y_hat_val[:, 20])
        sk_map8 = average_precision_score(np.sign(y_true_val[:, 21]).astype(int), y_hat_val[:, 21])

        sk_map = (sk_map1 + sk_map2 + sk_map3 + sk_map4 + sk_map5 + sk_map6 + sk_map7 + sk_map8) / 8.0

        self._data.append({
            'val_rocauc': sk_map,
        })
        return

    def get_data(self):
        return self._data

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

        return (bbx_pred)

    def Distance_parallel (self,y_true, y_pred):

        y_hat = tf.reshape(y_pred, [-1, 26])
        y = tf.cast(tf.reshape(y_true, [-1, 26]), tf.float32)

        c1 = tf.sqrt(tf.sign(y[:, 2]) *
                           (tf.square(y[:, 0] - y_hat[:, 0]) + tf.square(y[:, 1] - y_hat[:, 1])))

        c2 = tf.sqrt(tf.sign(y[:, 5]) *
                           (tf.square(y[:, 3] - y_hat[:, 3]) + tf.square(y[:, 4] - y_hat[:, 4])))

        c3 = tf.sqrt(tf.sign(y[:, 8]) *
                           (tf.square(y[:, 6] - y_hat[:, 6]) + tf.square(y[:, 7] - y_hat[:, 7])))

        c4 = tf.sqrt(tf.sign(y[:, 11]) *
                           (tf.square(y[:, 9] - y_hat[:, 9]) + tf.square(y[:, 10] - y_hat[:, 10])))

        c5 = tf.sqrt(tf.sign(y[:, 14]) *
                           (tf.square(y[:, 12] - y_hat[:, 12]) + tf.square(y[:, 13] - y_hat[:, 13])))

        c6 = tf.sqrt(tf.sign(y[:, 17]) *
                           (tf.square(y[:, 15] - y_hat[:, 15]) + tf.square(y[:, 16] - y_hat[:, 16])))

        c7 = tf.sqrt(tf.sign(y[:, 20]) *
                           (tf.square(y[:, 18] - y_hat[:, 18]) + tf.square(y[:, 19] - y_hat[:, 19])))

        c8 = tf.sqrt(tf.sign(y[:, 21]) *
                           (tf.square(y[:, 22] - y_hat[:, 22]) + tf.square(y[:, 23] - y_hat[:, 23])))

        wpar = [c1,c2,c3,c4,c5,c6,c7,c8]
        c1 = tf.reduce_mean(wpar[0])
        c2 = tf.reduce_mean(wpar[1])
        c3 = tf.reduce_mean(wpar[2])
        c4 = tf.reduce_mean(wpar[3])
        c5 = tf.reduce_mean(wpar[4])
        c6 = tf.reduce_mean(wpar[5])
        c7 = tf.reduce_mean(wpar[6])
        c8 = tf.reduce_mean(wpar[7])

        mean_distance = (c1+c2+c3+c4+c5+c6+c7+c8)/8.0

        # count1 = tf.cast(tf.count_nonzero(tf.sign(y[:, 2])), 'float')
        # count2 = tf.cast(tf.count_nonzero(tf.sign(y[:, 5])), 'float')
        # count3 = tf.cast(tf.count_nonzero(tf.sign(y[:, 8])), 'float')
        # count4 = tf.cast(tf.count_nonzero(tf.sign(y[:, 11])), 'float')
        # count5 = tf.cast(tf.count_nonzero(tf.sign(y[:, 14])), 'float')
        # count6 = tf.cast(tf.count_nonzero(tf.sign(y[:, 17])), 'float')
        # count7 = tf.cast(tf.count_nonzero(tf.sign(y[:, 20])), 'float')
        # summ = c1 + c2 + c3 + c4 + c5 + c6 + c7
        # count = count1 + count2 + count3 + count4 + count5 + count6 + count7
        # mean_distance = tf.where(tf.logical_not(tf.equal(count, 0)), x=(summ / tf.cast(count, 'float')), y=0)

        return mean_distance

    def AUC_all_parallel (self,y_true, y_pred):
        y_pred = tf.reshape(y_pred, [-1, 26])
        y_true = tf.keras.backend.sign(tf.reshape(y_true, [-1, 26]))

        auc_st1 = tf.metrics.auc(y_true[:, 2], y_pred[:, 2])
        auc_st2 = tf.metrics.auc(y_true[:, 5], y_pred[:, 5])
        auc_st3 = tf.metrics.auc(y_true[:, 8], y_pred[:, 8])
        auc_st4 = tf.metrics.auc(y_true[:, 11], y_pred[:, 11])
        auc_st5 = tf.metrics.auc(y_true[:, 14], y_pred[:, 14])
        auc_st6 = tf.metrics.auc(y_true[:, 17], y_pred[:, 17])
        auc_st7 = tf.metrics.auc(y_true[:, 20], y_pred[:, 20])

        auc_st = (auc_st1[1] + auc_st2[1] + auc_st3[1] + auc_st4[1] + auc_st5[1] + auc_st6[1] + auc_st7[1]) / 7

        return auc_st

    def new_acuracy_parallel(self,y_true, y_pred):

        conf = 0.5
        y_hat_re = tf.reshape(y_pred, [-1, 26])
        y_true_re = tf.reshape(y_true, [-1, 26])

        tp1 = tf.cast(tf.logical_and(tf.greater(y_hat_re[:, 2], conf), tf.cast(tf.sign(y_true_re[:, 2]), 'bool')),
                      'float')
        tp2 = tf.cast(tf.logical_and(tf.greater(y_hat_re[:, 5], conf), tf.cast(tf.sign(y_true_re[:, 5]), 'bool')),
                      'float')
        tp3 = tf.cast(tf.logical_and(tf.greater(y_hat_re[:, 8], conf), tf.cast(tf.sign(y_true_re[:, 8]), 'bool')),
                      'float')
        tp4 = tf.cast(tf.logical_and(tf.greater(y_hat_re[:, 11], conf), tf.cast(tf.sign(y_true_re[:, 11]), 'bool')),
                      'float')
        tp5 = tf.cast(tf.logical_and(tf.greater(y_hat_re[:, 14], conf), tf.cast(tf.sign(y_true_re[:, 14]), 'bool')),
                      'float')
        tp6 = tf.cast(tf.logical_and(tf.greater(y_hat_re[:, 17], conf), tf.cast(tf.sign(y_true_re[:, 17]), 'bool')),
                      'float')
        tp7 = tf.cast(tf.logical_and(tf.greater(y_hat_re[:, 20], conf), tf.cast(tf.sign(y_true_re[:, 20]), 'bool')),
                      'float')
        tp8 = tf.cast(tf.logical_and(tf.greater(y_hat_re[:, 21], conf), tf.cast(tf.sign(y_true_re[:, 21]), 'bool')),
                      'float')

        wpar = (y_hat_re, y_true_re, tp1, tp2, tp3, tp4, tp5, tp6, tp7, tp8)

        list = [2, 5, 8, 11, 14, 17, 20, 21]
        count_keypoints_totalcases = 0
        count_keypoints_pos_results = 0
        conf = 0.5
        for i in list:
            count_keypoints_totalcases += tf.cast(tf.count_nonzero(wpar[1][:, i]), 'float')
            count_keypoints_pos_results += tf.reduce_sum(tf.cast(tf.greater(wpar[0][:, i], conf), 'float'))

        summ = tf.reduce_sum(wpar[2]) + tf.reduce_sum(wpar[3]) + tf.reduce_sum(wpar[4]) + tf.reduce_sum(wpar[5]) + \
               tf.reduce_sum(wpar[6]) + tf.reduce_sum(wpar[7]) + tf.reduce_sum(wpar[8]) + tf.reduce_sum(wpar[9])
        RECAL = tf.where(tf.logical_not(tf.equal(count_keypoints_totalcases, 0)), x=(summ / count_keypoints_totalcases), y=0)
        PERCISION = tf.where(tf.logical_not(tf.equal(count_keypoints_pos_results, 0)), x=(summ / count_keypoints_pos_results), y=0)


        count_keypoints_totalcases_body = 0
        count_keypoints_pos_results_body = 0
        list1 = [2, 5, 8, 11, 14, 17, 20]
        for i in list1:
            count_keypoints_totalcases_body += tf.cast(tf.count_nonzero(wpar[1][:, i]), 'float')
            count_keypoints_pos_results_body += tf.reduce_sum(tf.cast(tf.greater(wpar[0][:, i], conf), 'float'))

        summ_body = tf.reduce_sum(wpar[2]) + tf.reduce_sum(wpar[3]) + tf.reduce_sum(wpar[4]) + tf.reduce_sum(wpar[5]) + \
               tf.reduce_sum(wpar[6]) + tf.reduce_sum(wpar[7]) + tf.reduce_sum(wpar[8])
        summ_detction = tf.reduce_sum(wpar[9])
        count_keypoints_totalcases_detection = tf.cast(tf.count_nonzero(wpar[1][:, 21]), 'float')
        count_keypoints_pos_results_detection = tf.reduce_sum(tf.cast(tf.greater(wpar[0][:, 21], conf), 'float'))
        re_body = tf.where(tf.logical_not(tf.equal(count_keypoints_totalcases, 0)), x=(summ_body / count_keypoints_totalcases_body), y=0)
        re_detection = tf.where(tf.logical_not(tf.equal(count_keypoints_totalcases, 0)), x=(summ_detction / count_keypoints_totalcases_detection), y=0)
        per_body = tf.where(tf.logical_not(tf.equal(count_keypoints_pos_results, 0)), x=(summ_body / count_keypoints_pos_results_body), y=0)
        per_detection = tf.where(tf.logical_not(tf.equal(count_keypoints_pos_results, 0)), x=(summ_detction / count_keypoints_pos_results_detection), y=0)

        # mAP1 = tf.metrics.average_precision_at_k(tf.cast(tf.sign(wpar[1][:, 2]),tf.int64), wpar[0][:,2], 1)[0]
        # mAP2 = tf.metrics.average_precision_at_k(tf.cast(tf.sign(wpar[1][:, 5]),tf.int64), wpar[0][:, 5], 1)[0]
        # mAP3 = tf.metrics.average_precision_at_k(tf.cast(tf.sign(wpar[1][:, 8]),tf.int64), wpar[0][:, 8], 1)[0]
        # mAP4 = tf.metrics.average_precision_at_k(tf.cast(tf.sign(wpar[1][:, 11]),tf.int64), wpar[0][:, 11], 1)[0]
        # mAP5 = tf.metrics.average_precision_at_k(tf.cast(tf.sign(wpar[1][:, 14]),tf.int64), wpar[0][:, 14], 1)[0]
        # mAP6 = tf.metrics.average_precision_at_k(tf.cast(tf.sign(wpar[1][:, 17]),tf.int64), wpar[0][:, 17], 1)[0]
        # mAP7 = tf.metrics.average_precision_at_k(tf.cast(tf.sign(wpar[1][:, 20]),tf.int64), wpar[0][:, 20], 1)[0]
        # mAP8 = tf.metrics.average_precision_at_k(tf.cast(tf.sign(wpar[1][:, 21]),tf.int64), wpar[0][:, 21], 1)[0]
        # mAP = tf.add_n([mAP1+mAP2+mAP3+mAP4+mAP5+mAP6+mAP7+mAP8])/8.0
        #mAP = 1
        return RECAL, PERCISION, re_body, re_detection, per_body, per_detection

    def RECAL(self,y_true, y_pred):

        RECAL, PERCISION, re_body, re_detection, per_body, per_detection  = self.new_acuracy_parallel(y_true, y_pred)
        return RECAL

    def PERCISION(self, y_true, y_pred):

        RECAL, PERCISION, re_body, re_detection, per_body, per_detection = self.new_acuracy_parallel(y_true, y_pred)
        return PERCISION

    def recall_body(self, y_true, y_pred):

        RECAL, PERCISION, re_body, re_detection, per_body, per_detection = self.new_acuracy_parallel(y_true, y_pred)
        return re_body

    def recall_detection(self, y_true, y_pred):

        RECAL, PERCISION, re_body, re_detection, per_body, per_detection = self.new_acuracy_parallel(y_true, y_pred)
        return re_detection

    def percision_body(self, y_true, y_pred):

        RECAL, PERCISION, re_body, re_detection, per_body, per_detection = self.new_acuracy_parallel(y_true, y_pred)
        return per_body

    def percision_detection(self, y_true, y_pred):

        RECAL, PERCISION, re_body, re_detection, per_body, per_detection = self.new_acuracy_parallel(y_true, y_pred)
        return per_detection

    def my_mAP_5(self, y_true, y_pred):

        _, _, map = self.new_acuracy_parallel(y_true, y_pred)
        return map

    def new_acuracy_parallel_8(self,y_true, y_pred):

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

        list = [2, 5, 8, 11, 14, 17, 20, 21]
        count_keypoints_totalcases = 0
        count_keypoints_pos_results = 0
        conf = 0.5
        for i in range(0, 8):
            count_keypoints_pos_results += tf.reduce_sum(tf.cast(tf.greater(wpar[0][:, i], conf), 'float'))
        for i in list:
            # count_keypoints_pos_results += tf.reduce_sum(tf.cast(tf.greater(wpar[0][:, i], conf), 'float'))
            count_keypoints_totalcases += tf.cast(tf.count_nonzero(wpar[1][:, i]), 'float')

        summ = tf.reduce_sum(wpar[2]) + tf.reduce_sum(wpar[3]) + tf.reduce_sum(wpar[4]) + tf.reduce_sum(wpar[5]) + \
               tf.reduce_sum(wpar[6]) + tf.reduce_sum(wpar[7]) + tf.reduce_sum(wpar[8]) + tf.reduce_sum(wpar[9])
        RECAL = tf.where(tf.logical_not(tf.equal(count_keypoints_totalcases, 0)), x=(summ / count_keypoints_totalcases),
                         y=-1)
        PERCISION = tf.where(tf.logical_not(tf.equal(count_keypoints_pos_results, 0)),
                             x=(summ / count_keypoints_pos_results), y=0)
        FP_rate = tf.where(tf.logical_not(tf.equal(count_keypoints_totalcases, 0)),
                           x=((count_keypoints_pos_results - summ) / count_keypoints_pos_results), y=0)

        mAP1 = tf.metrics.average_precision_at_k(tf.cast(tf.sign(wpar[1][:, 2]), tf.int64), wpar[0][:, 0], 1)[0]
        mAP2 = tf.metrics.average_precision_at_k(tf.cast(tf.sign(wpar[1][:, 5]), tf.int64), wpar[0][:, 1], 1)[0]
        mAP3 = tf.metrics.average_precision_at_k(tf.cast(tf.sign(wpar[1][:, 8]), tf.int64), wpar[0][:, 2], 1)[0]
        mAP4 = tf.metrics.average_precision_at_k(tf.cast(tf.sign(wpar[1][:, 11]), tf.int64), wpar[0][:, 3], 1)[0]
        mAP5 = tf.metrics.average_precision_at_k(tf.cast(tf.sign(wpar[1][:, 14]), tf.int64), wpar[0][:, 4], 1)[0]
        mAP6 = tf.metrics.average_precision_at_k(tf.cast(tf.sign(wpar[1][:, 17]), tf.int64), wpar[0][:, 5], 1)[0]
        mAP7 = tf.metrics.average_precision_at_k(tf.cast(tf.sign(wpar[1][:, 20]), tf.int64), wpar[0][:, 6], 1)[0]
        mAP8 = tf.metrics.average_precision_at_k(tf.cast(tf.sign(wpar[1][:, 21]), tf.int64), wpar[0][:, 7], 1)[0]
        mAP = tf.add_n([mAP1 + mAP2 + mAP3 + mAP4 + mAP5 + mAP6 + mAP7 + mAP8]) / 8.0

        return RECAL, PERCISION, mAP

    def RECAL_8(self,y_true, y_pred):

        re, _, _ = self.new_acuracy_parallel_8(y_true, y_pred)
        return re

    def PERCISION_8(self, y_true, y_pred):

        _, pe, _ = self.new_acuracy_parallel_8(y_true, y_pred)
        return pe

    def sk_mAP(self, parallel_model, batch_size, image_val, annotation_val, val_count):

        init_op = tf.global_variables_initializer()
        init_l = tf.local_variables_initializer()
        with tf.Session() as sess:
            sess.run(init_op)
            sess.run(init_l)

            sk_map = 0
            for bat_v in range(int(2693 / batch_size)):
                img_val, anno_val = sess.run([image_val, annotation_val])
                y_pred = parallel_model.predict(img_val, batch_size=batch_size)
                y_true_val = np.reshape(anno_val, [-1, 26])
                y_hat_val = np.reshape(y_pred, [-1, 26])
                sk_map1 = average_precision_score(np.sign(y_true_val[:, 2]).astype(int), y_hat_val[:, 2])
                sk_map2 = average_precision_score(np.sign(y_true_val[:, 5]).astype(int), y_hat_val[:, 5])
                sk_map3 = average_precision_score(np.sign(y_true_val[:, 8]).astype(int), y_hat_val[:, 8])
                sk_map4 = average_precision_score(np.sign(y_true_val[:, 11]).astype(int), y_hat_val[:, 11])
                sk_map5 = average_precision_score(np.sign(y_true_val[:, 14]).astype(int), y_hat_val[:, 14])
                sk_map6 = average_precision_score(np.sign(y_true_val[:, 17]).astype(int), y_hat_val[:, 17])
                sk_map7 = average_precision_score(np.sign(y_true_val[:, 20]).astype(int), y_hat_val[:, 20])
                sk_map8 = average_precision_score(np.sign(y_true_val[:, 21]).astype(int), y_hat_val[:, 21])

                sk_map += ((sk_map1 + sk_map2 + sk_map3 + sk_map4 + sk_map5 + sk_map6 + sk_map7 + sk_map8) / 8.0)

        return sk_map / int(val_count / batch_size)

    def get_max (self, y_true, y_pred):

        return tf.reduce_max(y_pred)

    def get_min(self, y_true, y_pred):

        return tf.reduce_min(y_pred)

    def sk_mAP_tensor(self, y_true, y_pred):

        y_true_val = tf.reshape(y_true, [-1, 26])
        y_hat_val = tf.reshape(y_pred, [-1, 26])
        sk_map1 = average_precision_score(np.sign(y_true_val[:, 2]).astype(int), y_hat_val[:, 2])
        sk_map2 = average_precision_score(np.sign(y_true_val[:, 5]).astype(int), y_hat_val[:, 5])
        sk_map3 = average_precision_score(np.sign(y_true_val[:, 8]).astype(int), y_hat_val[:, 8])
        sk_map4 = average_precision_score(np.sign(y_true_val[:, 11]).astype(int), y_hat_val[:, 11])
        sk_map5 = average_precision_score(np.sign(y_true_val[:, 14]).astype(int), y_hat_val[:, 14])
        sk_map6 = average_precision_score(np.sign(y_true_val[:, 17]).astype(int), y_hat_val[:, 17])
        sk_map7 = average_precision_score(np.sign(y_true_val[:, 20]).astype(int), y_hat_val[:, 20])
        sk_map8 = average_precision_score(np.sign(y_true_val[:, 21]).astype(int), y_hat_val[:, 21])

        sk_map = ((sk_map1 + sk_map2 + sk_map3 + sk_map4 + sk_map5 + sk_map6 + sk_map7 + sk_map8) / 8.0)

        return sk_map









