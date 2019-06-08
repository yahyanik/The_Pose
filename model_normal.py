from __future__ import division


'''
The custom model for light human pose detection
'''

import tensorflow as tf
import tensorflow.contrib as tc


class MobileNetV2_normal(object):

    def __init__(self, num_to_reduce=2, is_training=True, input_size=224, input_placeholder='give placeholder'):
        self.input_size = input_size
        self.is_training = is_training
        self.normalizer = tc.layers.batch_norm
        self.bn_params = {'is_training': self.is_training}

        with tf.variable_scope('MobileNetV2_custom'):
            self._create_placeholders(input_placeholder)
            self._build_model(num_to_reduce)
            # self.model = tf.keras.Model(inputs=X_input, outputs=X, name='HappyModel')

    def _create_placeholders(self, input_placeholder):
        # self.input = tf.placeholder(dtype=tf.float32, shape=[None, self.input_size, self.input_size, 3])
        self.input = input_placeholder

    def _build_model(self, num_to_reduce):
        self.i = 0
        with tf.variable_scope('init_conv'):
            output = tc.layers.conv2d(self.input, 32, 3, 2,
                                      normalizer_fn=self.normalizer, normalizer_params=self.bn_params)
        # print("shape after first layer", output.get_shape())
        self.output = self._inverted_bottleneck(output, 1, 16, 0)
        self.output = self._inverted_bottleneck(self.output, 6, 24, 1)  # 6 is the t and 34 is the output shape(finlter number and layers are repeated in for the n in the paper
        self.output = self._inverted_bottleneck(self.output, 6, 24, 0)
        self.output = self._inverted_bottleneck(self.output, 6, 32, 1)
        self.output = self._inverted_bottleneck(self.output, 6, 32, 0)
        self.output = self._inverted_bottleneck(self.output, 6, 32, 0)
        self.output = self._inverted_bottleneck(self.output, 6, 64, 1)
        self.output = self._inverted_bottleneck(self.output, 6, 64, 0)
        self.output = self._inverted_bottleneck(self.output, 6, 64, 0)
        self.output = self._inverted_bottleneck(self.output, 6, 64, 0)
        self.output = self._inverted_bottleneck(self.output, 6, 96, 0)
        self.output = self._inverted_bottleneck(self.output, 6, 96, 0)
        self.output = self._inverted_bottleneck(self.output, 6, 96, 0)
        self.output = self._inverted_bottleneck(self.output, 6, 160, 1)
        self.output = self._inverted_bottleneck(self.output, 6, 160, 0)
        self.output = self._inverted_bottleneck(self.output, 6, 160, 0)
        self.output = self._inverted_bottleneck(self.output, 6, 320, 0)
        num_to_reduce = int(num_to_reduce)
        # self.output = tc.layers.conv2d(self.output, 1280, 1, activation_fn=tf.nn.relu6, normalizer_fn=self.normalizer,
        #                                normalizer_params=self.bn_params)

        self.output = tc.layers.conv2d(self.output, num_to_reduce*26, 1, activation_fn=tf.nn.relu6, normalizer_fn=self.normalizer,
                                       normalizer_params=self.bn_params)
        # self.output = tc.layers.conv2d(self.output, 26, 1, activation_fn = None)

        # self.output = tc.layers.softmax(self.output)

        self.output = self.head_nework(self.output, num_to_reduce*24*100, num_to_reduce*2*100, num_to_reduce)
        #self.output = self.head_nework(self.output, 3692, 308, num_to_reduce)

    def _inverted_bottleneck(self, input, up_sample_rate, channels, subsample):
        with tf.variable_scope('inverted_bottleneck{}_{}_{}'.format(self.i, up_sample_rate, subsample)):
            self.i += 1
            stride = 2 if subsample else 1
            output = tc.layers.conv2d(input, up_sample_rate * input.get_shape().as_list()[-1], 1,
                                      activation_fn=tf.nn.relu6,
                                      normalizer_fn=self.normalizer, normalizer_params=self.bn_params)

            output = tc.layers.separable_conv2d(output, None, 3, 1, stride=stride,
                                                activation_fn=tf.nn.relu6,
                                                normalizer_fn=self.normalizer, normalizer_params=self.bn_params)

            output = tc.layers.conv2d(output, channels, 1, activation_fn=None,
                                      normalizer_fn=self.normalizer, normalizer_params=self.bn_params)

            if input.get_shape().as_list()[-1] == channels:
                output = tf.add(input, output)

            return output

    def head_nework(self, input, n_sigmoid, n_exp, num_to_reduce):
        with tf.variable_scope('head_network'):
            input_flat = tc.layers.flatten(input)
            keypoint_xy_class_probability = tc.layers.fully_connected(input_flat[:, :n_sigmoid],
                                                                  2400, activation_fn=tf.sigmoid)
            keypoint_xy_class_probability = tf.reshape(keypoint_xy_class_probability, [-1, 10, 10, 24])

            box_wh = tc.layers.fully_connected(input_flat[:, n_sigmoid:], 200, activation_fn=tf.exp)
            box_wh = tf.reshape(box_wh, [-1, 10, 10, 2])

            output = tf.concat([keypoint_xy_class_probability, box_wh], 3)
            output = tf.reshape(output, [-1, 10, 10, 26])
            return output

    def drop(self, input, rate, name):
        with tf.variable_scope('head_network{}'.format(name)):
            out = tc.layers.dropout(input, keep_prob=(1 - rate), is_training=self.is_training)

            return out





# def load_pb(pb):
#     with tf.gfile.GFile(pb, "rb") as f:
#         graph_def = tf.GraphDef()
#         graph_def.ParseFromString(f.read())
#     with tf.Graph().as_default() as graph:
#         tf.import_graph_def(graph_def, name='')
#         return graph
#
# # ***** (1) Create Graph *****
#
# g = tf.Graph()
# # g = MobileNetV2_1280relu_new(2,False, 320)
#
# sess = tf.Session(graph=g)
#
# with g.as_default():
#     x = tf.placeholder(dtype=tf.float32, shape=[None, 320, 320, 3], name='x')
#
#     A = MobileNetV2_1280relu_new(2, False, 320,x)
#     res = tf.identity(A.output, name="output")
#     sess.run(tf.global_variables_initializer())
#     flops = tf.profiler.profile(g, options = tf.profiler.ProfileOptionBuilder.float_operation())
#     print('FLOP before freezing', flops.total_float_ops)
# # *****************************
#
# # # ***** (2) freeze graph *****
# output_graph_def = tf.graph_util.convert_variables_to_constants(sess, g.as_graph_def(), ['output'])
#
# with tf.gfile.GFile('graph.pb', "wb") as f:
#     f.write(output_graph_def.SerializeToString())
# # *****************************
#
#
# # ***** (3) Load frozen graph *****
# g2 = load_pb('./graph.pb')
# with g2.as_default():
#     flops = tf.profiler.profile(g2, options = tf.profiler.ProfileOptionBuilder.float_operation())
#     print('FLOP after freezing', flops.total_float_ops)
#
#
#
#
# g = tf.train.import_meta_graph('/home/yahya/programming/workplace//model_relu_7/tmp/model.ckpt-51.meta')
# # g = MobileNetV2_1280relu_new(2,False, 320)
#
# sess = tf.Session()
# tf.train.Saver.restore(sess, './model_relu_7/tmp/model.ckpt-51.data-00000-of-00001')
#
# tf.graph_util.convert_variables_to_constants(sess,sess.graph.as_graph_def(),['output'])




# import tensorflow as tf
# from tensorflow.python.framework import graph_util
#
# def load_pb(pb):
#     with tf.gfile.GFile(pb, "rb") as f:
#         graph_def = tf.GraphDef()
#         graph_def.ParseFromString(f.read())
#     with tf.Graph().as_default() as graph:
#         tf.import_graph_def(graph_def, name='')
#         return graph
#
# # ***** (1) Create Graph *****
# g = tf.Graph()
# sess = tf.Session(graph=g)
# with g.as_default():
#     A = tf.Variable(initial_value=tf.random_normal([25, 16]))
#     B = tf.Variable(initial_value=tf.random_normal([16, 9]))
#     C = tf.matmul(A, B)
#     D = tf.nn.sigmoid(C)
#     E = tf.layers.batch_normalization(D,axis=-1,name = 'out')
#
#     sess.run(tf.global_variables_initializer())
#     flops = tf.profiler.profile(g, options = tf.profiler.ProfileOptionBuilder.float_operation())
#     print('FLOP before freezing', flops.total_float_ops)
# # *****************************
#
# # ***** (2) freeze graph *****
# output_graph_def = graph_util.convert_variables_to_constants(sess, g.as_graph_def(), ['out'])
#
# with tf.gfile.GFile('graph.pb', "wb") as f:
#     f.write(output_graph_def.SerializeToString())
# # *****************************
#
#
# # ***** (3) Load frozen graph *****
# g2 = load_pb('./graph.pb')
# with g2.as_default():
#     flops = tf.profiler.profile(g2, options = tf.profiler.ProfileOptionBuilder.float_operation())
#     print('FLOP after freezing', flops.total_float_ops)