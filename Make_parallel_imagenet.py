from __future__ import division
import tensorflow as tf
from Metrics_8 import *
from model_8 import *
from cost_8 import *

'''
The following functions make everything run in the Train file parallel, These functions split everything for each batch
and run the functions and concat the results. The model_cost_function should have all the functions you need to run 
The make_parallel breaks the model_cost_function and runs it on GPUs and concat them at the end and gives back to the train module.
'''




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
    MODEL_OUTPUT = []
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
                MODEL_OUTPUT.append([model.output])
                metric_obj = (out_split_fn)[6]
                out_aplit_new_acuracy.append(out_split_fn[7])

    list_metrix = []
    me_metrix = []
    accuracy_all_metrix = []
    au_detection_metrix= []
    au_class_metrix= []
    new_acuracy_matrix = []
    MODEL_OUTPUT_MATRIX = []

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
    for wpar in (range(len(MODEL_OUTPUT[0]))):
        MODEL_OUTPUT_MATRIX.append(tf.concat([MODEL_OUTPUT[z][wpar] for z in range(len(MODEL_OUTPUT))], axis=0))

    return list_metrix, model, me_metrix, accuracy_all_metrix, au_detection_metrix, au_class_metrix, metric_obj, new_acuracy_matrix,MODEL_OUTPUT_MATRIX