from __future__ import division

from tensorflow.python.platform import gfile
from model_normal_keras import *
from Metrics_keras import *
from cost_keras import *
from tensorflow.keras import backend as K

# K.set_learning_phase(0)

'''
The first part of this code is to freeze the model after training and to be able to run it in low level APIs and the second part is to count the FLOPs needed
'''


# def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
#     from tensorflow.python.framework.graph_util import convert_variables_to_constants
#     graph = session.graph
#     with graph.as_default():
#         freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
#         output_names = output_names or []
#         output_names += [v.op.name for v in tf.global_variables()]
#         # Graph -> GraphDef ProtoBuf
#         input_graph_def = graph.as_graph_def()
#         if clear_devices:
#             for node in input_graph_def.node:
#                 node.device = ""
#         frozen_graph = convert_variables_to_constants(session, input_graph_def,
#                                                       output_names, freeze_var_names)
#         return frozen_graph
#
#
# g = tf.Graph()
# sess = tf.Session(graph=g)
#
# # ***** freeze graph *****
# model_obj = MobileNetV2_normal_keras(num_to_reduce=32, drop_fac=0.6, head_is_training=False,
#                                          regular_fac=0.1,
#                                          layers_to_fine_tune=155, include_top=False, train_layers=False)
# metric = metric_custom()
# metric_list = [metric.RECAL, metric.PERCISION, metric.Distance_parallel, metric.get_max, metric.get_min,
#                    metric.recall_body, metric.percision_body, metric.recall_detection, metric.percision_detection]
#
# model_obj.model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0001),loss=my_cost_MSE, metrics=metric_list)
# frozen_graph = freeze_session(K.get_session(),output_names=[out.op.name for out in model_obj.model.outputs])
# #
# tf.train.write_graph(frozen_graph, "./frozen", "my_model.pb", as_text=False)
#
#
# f = gfile.FastGFile("./frozen/my_model.pb", 'rb')
# graph_def = tf.GraphDef()
# # Parses a serialized binary message into the current message.
# graph_def.ParseFromString(f.read())
# f.close()
#
# sess.graph.as_default()
# # Import a serialized TensorFlow `GraphDef` protocol buffer
# # and place into the current default `Graph`.
# tf.import_graph_def(graph_def)



import tensorflow as tf
import tensorflow.keras.backend as K


def get_flops(model):
    run_meta = tf.RunMetadata()
    opts = tf.profiler.ProfileOptionBuilder.float_operation()

    # We use the Keras session graph in the call to the profiler.
    flops = tf.profiler.profile(graph=K.get_session().graph,
                                run_meta=run_meta, cmd='op', options=opts)

    return flops.total_float_ops  # Prints the "flops" of the model.


# .... Define your model here ....

# base_model = tf.keras.applications.MobileNetV2(include_top =False ,alpha=1, weights=None, input_tensor=tf.keras.layers.Input(shape=(320, 320, 3)))
# flat = tf.keras.layers.AveragePooling2D(pool_size=(10, 10), strides=None, padding='valid', data_format=None)(tf.keras.layers.Input(shape=(10, 10, 1280)))
# flat = tf.keras.layers.Flatten()(flat)
#
# dense0 = tf.keras.layers.Dense(1024, activation='relu')(flat)
#
# out0 = tf.keras.layers.Dense(2400, activation='sigmoid')(dense0)
#
# dense1 = tf.keras.layers.Dense(64, activation='relu')(flat)
# out1 = tf.keras.layers.Dense(200, activation=None)(dense1)
#
# out_flaten = tf.keras.layers.concatenate([out0, out1], axis=-1)
# output = tf.keras.layers.Reshape((10, 10, 26))(out_flaten)
# model1 = tf.keras.models.Model(name='Keypoint_Detection1', inputs=tf.keras.layers.Input(shape=(10, 10, 1280)), outputs=[output])

model_obj = MobileNetV2_normal_keras(num_to_reduce=32, drop_fac=0.6, head_is_training=False,
                                         regular_fac=0.1,
                                         layers_to_fine_tune=5, include_top=False, train_layers=False)

# print(get_flops(model_obj.model))
print(get_flops(model_obj.model))