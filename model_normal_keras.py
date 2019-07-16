from __future__ import division


'''
The Keras implementation on Mobilenet for transfer learning
'''

import tensorflow as tf
import tensorflow.keras as K
import tensorflow.contrib as tc
from tensorflow.keras import regularizers

class MobileNetV2_normal_keras(object):

    def __init__(self, num_to_reduce=2, head_is_training=True, regular_fac=0.01,layers_to_fine_tune=100, include_top = True, fireezed_layers=True):

        IMG_SHAPE = (224, 224, 3)
        self.head_is_training = head_is_training
        self.regular_fac = regular_fac
        self.layers_to_fine_tune=layers_to_fine_tune
        self._build_model(num_to_reduce, IMG_SHAPE, include_top, fireezed_layers)



    def _build_model(self, num_to_reduce, IMG_SHAPE, include_top, fireezed_layers):
        input_tensor = K.layers.Input(shape=(320, 320, 3))
        base_model = K.applications.MobileNetV2(input_tensor=input_tensor,
                                                       include_top=include_top,
                                                       weights='imagenet')
        base_model.trainable = self.head_is_training
        # print (len(base_model.layers))
        num_to_reduce = int(num_to_reduce)
        self._head_nework(base_model, num_to_reduce*24*100)
        self.model = K.models.Model(name='Keypoint_Detection', inputs=base_model.input, outputs=[self.output])
        # self.model= K.Sequential([base_model])
        self.model.layers[0] = input_tensor
        for layer in self.model.layers[:self.layers_to_fine_tune]:
            layer.trainable = fireezed_layers

    def _head_nework(self, base_model, split_point):

        '''
        flat = K.layers.Flatten()(base_model.output)
        dense0 = K.layers.Dense(1600, activation=tf.nn.leaky_relu, kernel_regularizer=regularizers.l2(self.regular_fac))(flat)
        out0 = K.layers.Dense(2400, activation=tf.sigmoid)(dense0)
        dense1 = K.layers.Dense(160, activation=tf.nn.leaky_relu, kernel_regularizer=regularizers.l2(self.regular_fac))(flat)
        out1 = K.layers.Dense(200, activation=tf.sigmoid)(dense1)
        out_flaten = K.layers.concatenate([out0, out1], axis=-1)
        self.output = K.layers.Reshape((10, 10, 26))(out_flaten)
        '''

        flat = K.layers.Flatten()(base_model.output)
        dense0 = K.layers.Dense(3200, activation=tf.nn.leaky_relu,
                                kernel_regularizer=regularizers.l2(self.regular_fac))(flat)
        dense1 = K.layers.Dense(1600, activation=tf.nn.leaky_relu,
                                kernel_regularizer=regularizers.l2(self.regular_fac))(dense0)
        out0 = K.layers.Dense(800, activation=tf.sigmoid)(dense1)
        # dense1 = K.layers.Dense(160, activation=tf.nn.leaky_relu, kernel_regularizer=regularizers.l2(self.regular_fac))(
        #     flat)
        # out1 = K.layers.Dense(200, activation=tf.sigmoid)(dense1)
        # out_flaten = K.layers.concatenate([out0, out1], axis=-1)
        self.output = K.layers.Reshape((10, 10, 8))(out0)


