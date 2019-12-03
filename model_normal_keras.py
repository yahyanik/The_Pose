from __future__ import division


'''
The Keras implementation on Mobilenet for transfer learning
'''

import tensorflow as tf
import tensorflow.keras as K
import tensorflow.contrib as tc
from tensorflow.keras import regularizers

class MobileNetV2_normal_keras(object):

    def __init__(self, num_to_reduce=2, drop_fac = 1, head_is_training=True, regular_fac=0.01,layers_to_fine_tune=100, include_top = True, train_layers=True):

        #IMG_SHAPE = (224, 224, 3)
        self.head_is_training = head_is_training
        self.drop_fac = drop_fac
        self.regular_fac = regular_fac
        # self.input_tensor = tf.placeholder('float32', shape=(1,320,320,3)) # use this line for the Flops question
        self.input_tensor = K.layers.Input(shape=(320, 320, 3))
        self.layers_to_fine_tune=layers_to_fine_tune
        self._build_model(num_to_reduce, include_top, train_layers)




    def _build_model(self, num_to_reduce, include_top, train_layers):
        input_tensor = K.layers.Input(shape=(320, 320, 3))
        base_model = K.applications.MobileNetV2(input_tensor=self.input_tensor,
                                                       include_top=include_top,
                                                       weights='imagenet') # not including top
        for layer in base_model.layers[:self.layers_to_fine_tune]:
            layer.trainable = train_layers
        print (base_model.summary())
        # print ('len(basemodel.trainable_variables)', len(base_model.layers))
        # num_to_reduce = int(num_to_reduce)
        self._head_nework(base_model, 1180)     #num_to_reduce*24*100
        self.model = K.models.Model(name='Keypoint_Detection', inputs=self.input_tensor, outputs=[self.output])
        # self.model= K.Sequential([base_model])
        # self.model.layers[0] = input_tensor


    def _head_nework(self, base_model, split_point):

        # flat = K.layers.Dropout((1 - self.drop_fac))(base_model.output)
        # flat = K.layers.Conv2D(1024, (1, 1), activation='relu')(flat)
        # flat = K.layers.Dropout((1 - self.drop_fac))(flat)
        # flat = K.layers.Conv2D(512, (1, 1), activation='relu')(flat)

        flat = K.layers.AveragePooling2D(pool_size=(10, 10), strides=None, padding='valid', data_format=None)(base_model.output)
        flat = K.layers.Flatten()(flat)
        # flat = K.layers.Flatten()(base_model.output)
        flat = K.layers.BatchNormalization()(flat)
        # print(self.drop_fac)
        flat = K.layers.Dropout((1-self.drop_fac))(flat)

        dense0 = K.layers.Dense(1024, activation='relu', kernel_regularizer=regularizers.l1_l2(self.regular_fac,self.regular_fac))(flat)
        # dense0 = K.layers.Dense(1000, activation='relu')(flat)
        # dense0 = K.layers.BatchNormalization()(dense0)
        # dense1 = K.layers.Dense(1000, activation='relu', kernel_regularizer=regularizers.l1(self.regular_fac))(dense0)
        # dense1 = K.layers.BatchNormalization()(dense1)
        out0 = K.layers.Dense(2400, activation='sigmoid')(dense0)
        # out0 = K.layers.Conv2D(24, (1,1), activation='sigmoid')(flat)

        # dense11 = K.layers.Dense(400, activation='relu', kernel_regularizer=regularizers.l1_l2(self.regular_fac,self.regular_fac))(flat)
        # dense11 = K.layers.Dense(400, activation='relu')(flat)

        #-p[;''''''''''''''dense11 = K.layers.BatchNormalization()(dense11)
        dense1 = K.layers.Dense(64, activation='relu',kernel_regularizer=regularizers.l1(self.regular_fac))(flat)
        out1 = K.layers.Dense(200, activation=None)(dense1)

        out_flaten = K.layers.concatenate([out0, out1], axis=-1)
        self.output = K.layers.Reshape((10, 10, 26))(out_flaten)

        # self.output= K.layers.concatenate([out0, out1], axis=-1)






        # flat = K.layers.Flatten()(base_model.output)
        # dense0 = K.layers.Dense(1600, activation=tf.nn.leaky_relu)(flat)
        # dense1 = K.layers.Dense(1600, activation=tf.nn.leaky_relu)(dense0)
        # dense2 = K.layers.Dense(800, activation=tf.nn.leaky_relu)(dense1)
        # out0 = K.layers.Dense(800, activation=tf.sigmoid)(dense2)
        # self.output = K.layers.Reshape((10, 10, 8))(out0)


#todo :with batchnorm above and 1000 in the next and l1 (77) is bettter than l2(78). we need more regularization too. l2 gives 92 and 73 which means its ok, we need no more nodes.
#todo: I am putting 79 to l1_l2 with the arch: batchnorm and 1000 and sigmoid from 117.
#todo: the 78 is with l2 and batchonorm and 1000 and 144 nodes pre-trained and heavy over trained.
#todo: the 77 is l1 and 1000 and 2400 and 144 and fairly over fit, until epoch 15 is not bad. I need to test the 155 too, instead of 144
#todo: with the model that is in 79 that has l1 and l2 with 1000 and 2400 and 400 and 200 and no batchnorm about 30 steps are good which are more than before. 0.05 is the regularization factor
#todo: in model 80 I have batchnorm -> 1000 to 2400 and 400 to 200 with l1 and l2 and fac 0.1 with batchnorm after each layer I got the best of 67 and ~28
#todo: in model 80 I have batchnorm -> 1000 to 2400 and 400 to 200 with l1 and l2 and fac 0.1 with batchnorm after each layer I got the best of 53 and ~14 with 156.
#todo; the 81 model is the same as the 80 with 0.01 for regularization and 0.9 drop out rate and with 117 to not train. per 0.31 and recall of 0.65.
#todo: in 82 with 0.8 dropout and 0.05 regularization. I got 28 per va 66 recall.
#todo: in 83 the same is happening like 81 with 144 instead of 117. we got 63 and 22.
#todo: model 84 is the same with 91 not trained. 67 recall and pre 29.
#todo: 85 is 91 nontraining with 1000 and 1000  with 0.8 drop and .05 reg. recall 66 and per of 28.
#todo: 86 with 91 and 0.08 and 0.7 has 63 and 30.
#todo: 89 with conv2d ba 512 neuron ba 0.7 va 0.1 has the recal 40 and percision 40 as well
#todo: 90 ith conv2d ba 256 neurons ba 0.7 va 0.1 has the recall 36 va percision 40 a well
#todo: 92 with conv2d 4096 0.8 va 0.005 53 20
#todo: 93 qith 2 4096 0.8 va 0.005 53 va 22
#todo: 94 with 1024 0.7 va 0.05 53 vaf 22
#todo: log and MSE gives nan
#todo: 95 with 117 1024 0.7 va 0.01 53 vaf 18 ba log
#todo: 96 with 144 1024 0.6 va 0.01 36 vaf 11 ba log
#todo: with 256 node recall 51 and pre is 18


