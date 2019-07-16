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
        # self.output = self._inverted_bottleneck(output, 1, 32, 0)
        self.output = self._inverted_bottleneck(self.output, 6, 24, 1)  # 6 is the t and 34 is the output shape(finlter number and layers are repeated in for the n in the paper
        # self.output = self._inverted_bottleneck(self.output, 6, 48, 1)
        self.output = self._inverted_bottleneck(self.output, 6, 24, 0)
        # self.output = self._inverted_bottleneck(self.output, 6, 48, 0)
        self.output = self._inverted_bottleneck(self.output, 6, 32, 1)
        # self.output = self._inverted_bottleneck(self.output, 6, 64, 1)
        self.output = self._inverted_bottleneck(self.output, 6, 32, 0)
        # self.output = self._inverted_bottleneck(self.output, 6, 64, 0)
        # self.output = self._inverted_bottleneck(self.output, 6, 32, 0)
        # self.output = self._inverted_bottleneck(self.output, 6, 64, 0)
        self.output = self._inverted_bottleneck(self.output, 6, 64, 1) #######!!!!!!!!!!!!!! 1 changed to 0
        # self.output = self._inverted_bottleneck(self.output, 6, 128, 1)
        self.output = self._inverted_bottleneck(self.output, 6, 64, 0)
        # self.output = self._inverted_bottleneck(self.output, 6, 128, 0)
        self.output = self._inverted_bottleneck(self.output, 6, 64, 0)
        # self.output = self._inverted_bottleneck(self.output, 6, 128, 0)
        self.output = self._inverted_bottleneck(self.output, 6, 64, 0)
        # self.output = self._inverted_bottleneck(self.output, 6, 128, 0)
        self.output = self._inverted_bottleneck(self.output, 6, 96, 0)
        # self.output = self._inverted_bottleneck(self.output, 6, 192, 0)
        self.output = self._inverted_bottleneck(self.output, 6, 96, 0)
        # self.output = self._inverted_bottleneck(self.output, 6, 192, 0)
        self.output = self._inverted_bottleneck(self.output, 6, 96, 0)
        # self.output = self._inverted_bottleneck(self.output, 6, 192, 0)
        self.output = self._inverted_bottleneck(self.output, 6, 160, 1)
        # self.output = self._inverted_bottleneck(self.output, 6, 320, 1)
        self.output = self._inverted_bottleneck(self.output, 6, 160, 0)
        # self.output = self._inverted_bottleneck(self.output, 6, 320, 0)
        self.output = self._inverted_bottleneck(self.output, 6, 160, 0)
        # self.output = self._inverted_bottleneck(self.output, 6, 320, 0)
        self.output = self._inverted_bottleneck(self.output, 6, 320, 0)
        # self.output = self._inverted_bottleneck(self.output, 6, 640, 0)
        num_to_reduce = int(num_to_reduce)
        # print (self.output.shape)
        self.output = tc.layers.conv2d(self.output, 640, 1, activation_fn=tf.nn.relu6, normalizer_fn=self.normalizer,
                                       normalizer_params=self.bn_params)

        # self.output = tc.layers.conv2d(self.output, num_to_reduce*8, 1, activation_fn=tf.nn.relu6, normalizer_fn=self.normalizer,
        #                                normalizer_params=self.bn_params)
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

            '''
            THE ABSENCE OF THE PARAMETERS IN THE FULLY CONNECTED LAYTER AT THE END MAKES THE NETWORK FROM VERY BAD TO VERY GOOD.
            '''

            output = tc.layers.fully_connected(input_flat, 320, activation_fn=tf.nn.leaky_relu)
            output = tc.layers.fully_connected(output, 320, activation_fn=None)
            # output = tc.layers.fully_connected(output, 3200, activation_fn=None)

            keypoint_xy_class_probability = tc.layers.fully_connected(output, 800, activation_fn=tf.sigmoid)
            output = tf.reshape(keypoint_xy_class_probability, [-1, 10, 10, 8])

            # box_wh = tc.layers.fully_connected(input_flat[:, n_sigmoid:], 200, activation_fn=tf.exp)
            # box_wh = tf.reshape(box_wh, [-1, 10, 10, 2])
            #
            # output = tf.concat([keypoint_xy_class_probability, box_wh], 3)
            # output = tf.reshape(output, [-1, 10, 10, 26])
            return output

    def drop(self, input, rate, name):
        with tf.variable_scope('head_network{}'.format(name)):
            out = tc.layers.dropout(input, keep_prob=(1 - rate), is_training=self.is_training)

            return out