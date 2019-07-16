import tensorflow as tf
import tensorflow.contrib as tc

class MobileNetV2(object):

    def __init__(self, is_training=True, input_size=224, input_placeholder=tf.keras.Input((224, 224, 3))):
        self.input_size = input_size
        self.is_training = is_training
        self.normalizer = tc.layers.batch_norm
        self.bn_params = {'is_training': self.is_training}

        with tf.variable_scope('MobileNetV2'):
            self._create_placeholders(input_placeholder)
            self._build_model()
            # self.model = tf.keras.Model(inputs=X_input, outputs=X, name='HappyModel')

    def _create_placeholders(self, input_placeholder):
        # self.input = tf.placeholder(dtype=tf.float32, shape=[None, self.input_size, self.input_size, 3])
        self.input = input_placeholder


    def _build_model(self):
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

        self.output = tc.layers.conv2d(self.output, 1280, 1, activation_fn = tf.nn.relu, normalizer_fn=self.normalizer, normalizer_params=self.bn_params)
        self.output = tc.layers.avg_pool2d(self.output, 7)
        self.output = tc.layers.conv2d(self.output, 1000, 1, activation_fn=None)

    def _inverted_bottleneck(self, input, up_sample_rate, channels, subsample):
        with tf.variable_scope('inverted_bottleneck{}_{}_{}'.format(self.i, up_sample_rate, subsample)):
            self.i += 1
            stride = 2 if subsample else 1
            output = tc.layers.conv2d(input, up_sample_rate*input.get_shape().as_list()[-1], 1,
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


if __name__ == '__main__':
    model = MobileNetV2( is_training=True, input_size=224, input_placeholder=tf.keras.Input((224, 224, 3)))

    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, '/home/yahya/programing/workspace/The_Pose/pre-trained/mobilenet_v2_1.4_224.ckpt')
        # new_saver.restore(sess, tf.train.latest_checkpoint('./'))