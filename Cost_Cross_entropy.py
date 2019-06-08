from __future__ import division
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard

def CPU_reminder_of_cost(wpar, y_true, batch_size, regular_fac):

    all = 1                 # 0.009528889   between zeros and 1s
    landa = 5             # 0.990561053   between all and zeros
    lada = (503705/65088000)      # 0.009438946   between all and 1s
    gamma = 1 - lada          #next change to 5,0.5,1
    bbx_factor = 5

    c1 = tf.reduce_sum(wpar[0], axis=-1)
    c2 = tf.reduce_sum(wpar[1], axis=-1)
    c3 = tf.reduce_sum(wpar[2], axis=-1)
    c4 = tf.reduce_sum(wpar[3], axis=-1)
    c5 = tf.reduce_sum(wpar[4], axis=-1)
    c6 = tf.reduce_sum(wpar[5], axis=-1)
    c7 = tf.reduce_sum(wpar[6], axis=-1)

    v1 = tf.reduce_sum(wpar[7], axis=-1)
    v2 = tf.reduce_sum(wpar[8], axis=-1)
    v3 = tf.reduce_sum(wpar[9], axis=-1)
    v4 = tf.reduce_sum(wpar[10], axis=-1)
    v5 = tf.reduce_sum(wpar[11], axis=-1)
    v6 = tf.reduce_sum(wpar[12], axis=-1)
    v7 = tf.reduce_sum(wpar[13], axis=-1)
    v8 = tf.reduce_sum(wpar[14], axis=-1)

    v11 = tf.reduce_sum(wpar[15], axis=-1)
    v21 = tf.reduce_sum(wpar[16], axis=-1)
    v31 = tf.reduce_sum(wpar[17], axis=-1)
    v41 = tf.reduce_sum(wpar[18], axis=-1)
    v51 = tf.reduce_sum(wpar[19], axis=-1)
    v61 = tf.reduce_sum(wpar[20], axis=-1)
    v71 = tf.reduce_sum(wpar[21], axis=-1)
    v81 = tf.reduce_sum(wpar[22], axis=-1)

    b1 = tf.reduce_sum(wpar[23], axis=-1)
    b2 = tf.reduce_sum(wpar[24], axis=-1)

    f1 = (landa * (c1 + c2 + c3 + c4 + c5 + c6 + c7 + b1))
    f2 = (gamma * (v11+v21+v31+v41+v51+v61+v71+v81))
    f3 = (lada * (v1+v2+v3+v4+v5+v6+v7+v8))
    f4 = (bbx_factor * b2)
    # cost_general = (f1+f2+f3+f4)
    cost_general = f2+f3

    m = [tf.nn.l2_loss(tf.cast(v, tf.float32))
                                      for v in tf.trainable_variables() if 'bias' not in v.name]

    l2_loss = regular_fac * tf.add_n(m)
    # l2_loss = tc.layers.apply_regularization(tc.layers.l2_regularizer(regular_fac), weights_list = None)
    # l2_loss = 0

    cost_total = (cost_general+l2_loss)/batch_size

    return cost_total,f1,f2,l2_loss,f3


def cost (y_pred, y_true):

    y = tf.reshape(y_true, [-1, 26])
    y_hat = tf.reshape(y_pred, [-1, 26])

    c1 = tf.sign(y[:,2]) * (
            tf.squared_difference(y[:,0] , y_hat[:,0]) +
            tf.squared_difference(y[:,1] , y_hat[:,1]))

    c2 = tf.sign(y[:, 5]) * (
            tf.squared_difference(y[:, 3] , y_hat[:, 3]) +
            tf.squared_difference(y[:, 4] , y_hat[:, 4]))

    c3 = tf.sign(y[:, 8]) * (
            tf.squared_difference(y[:, 6] , y_hat[:, 6]) +
            tf.squared_difference(y[:, 7] , y_hat[:, 7]))

    c4 = tf.sign(y[:, 11]) * (
            tf.squared_difference(y[:, 10] , y_hat[:, 10]) +
            tf.squared_difference(y[:, 9] , y_hat[:, 9]))

    c5 = tf.sign(y[:, 14]) * (
            tf.squared_difference(y[:, 12] , y_hat[:, 12]) +
            tf.squared_difference(y[:, 13] , y_hat[:, 13]))

    c6 = tf.sign(y[:, 17]) * (
            tf.squared_difference(y[:, 15] , y_hat[:, 15]) +
            tf.squared_difference(y[:, 16] , y_hat[:, 16]))

    c7 = tf.sign(y[:, 20]) * (
            tf.squared_difference(y[:, 18] , y_hat[:, 18]) +
            tf.squared_difference(y[:, 19] , y_hat[:, 19]))

    v1 = (-1) * ((-1) * tf.sign(y[:, 2]) + 1) * (tf.log(1-y_hat[:, 2]))
    v2 = (-1) * ((-1) * tf.sign(y[:, 5]) + 1) * (tf.log(1-y_hat[:, 5]))
    v3 = (-1) * ((-1) * tf.sign(y[:, 8]) + 1) * (tf.log(1-y_hat[:, 8]))
    v4 = (-1) * ((-1) * tf.sign(y[:, 11]) + 1) * (tf.log(1-y_hat[:, 11]))
    v5 = (-1) * ((-1) * tf.sign(y[:, 14]) + 1) * (tf.log(1-y_hat[:, 14]))
    v6 = (-1) * ((-1) * tf.sign(y[:, 17]) + 1) * (tf.log(1-y_hat[:, 17]))
    v7 = (-1) * ((-1) * tf.sign(y[:, 20]) + 1) * (tf.log(1-y_hat[:, 20]))
    v8 = (-1) * ((-1) * tf.sign(y[:, 21]) + 1) * (tf.log(1-y_hat[:, 21]))

    v11 = (-1) * tf.sign(y[:, 2]) * (tf.log(1-y_hat[:, 2]))       # in model_1280relu_7 series there is no sign() for the
    v21 = (-1) * tf.sign(y[:, 5]) * (tf.log(1-y_hat[:, 5]))   # y_hat in the error calculation just one for before the error
    v31 = (-1) * tf.sign(y[:, 8]) * (tf.log(1-y_hat[:, 8]))
    v41 = (-1) * tf.sign(y[:, 11]) * (tf.log(1-y_hat[:, 11]))
    v51 = (-1) * tf.sign(y[:, 14]) * (tf.log(1-y_hat[:, 14]))
    v61 = (-1) * tf.sign(y[:, 17]) * (tf.log(1-y_hat[:, 17]))
    v71 = (-1) * tf.sign(y[:, 20]) * (tf.log(1-y_hat[:, 20]))
    v81 = (-1) * tf.sign(y[:, 21]) * (tf.log(1-y_hat[:, 21]))

    b1 = tf.sign(y[:, 21]) * (
            tf.squared_difference(y[:, 22] , y_hat[:, 22]) +
            tf.squared_difference(y[:, 23] , y_hat[:, 23]))

    b2 = tf.sign(y[:, 21]) * (
            tf.squared_difference(tf.sqrt(y[:, 24]) , tf.sqrt(y_hat[:, 24])) +
            tf.squared_difference(tf.sqrt(y[:, 25]) , tf.sqrt(y_hat[:, 25])))

    cost = (c1,c2,c3,c4,c5,c6,c7,v1,v2,v3,v4,v5,v6,v7,v8,v11,v21,v31,v41,v51,v61,v71,v81,b1,b2)
    return cost

