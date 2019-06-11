from __future__ import division
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard


'''
The tailored cost function for the Human Gesture detection. The cost function has two parts
The cost runs at GPU and calculates all the value and the CPU_reminder_of_cost runs after that to put every thing together
The CPU_reminder_of_cost function runs at the CPU and the rest are parallel GPU implementation. 
'''

def CPU_reminder_of_cost(wpar, y_true, batch_size, regular_fac):


    lada = (503705/65088000)
    gamma = 1 - lada

    v1 = tf.reduce_sum(wpar[0], axis=-1)
    v2 = tf.reduce_sum(wpar[1], axis=-1)
    v3 = tf.reduce_sum(wpar[2], axis=-1)
    v4 = tf.reduce_sum(wpar[3], axis=-1)
    v5 = tf.reduce_sum(wpar[4], axis=-1)
    v6 = tf.reduce_sum(wpar[5], axis=-1)
    v7 = tf.reduce_sum(wpar[6], axis=-1)
    v8 = tf.reduce_sum(wpar[7], axis=-1)

    v11 = tf.reduce_sum(wpar[8], axis=-1)
    v21 = tf.reduce_sum(wpar[9], axis=-1)
    v31 = tf.reduce_sum(wpar[10], axis=-1)
    v41 = tf.reduce_sum(wpar[11], axis=-1)
    v51 = tf.reduce_sum(wpar[12], axis=-1)
    v61 = tf.reduce_sum(wpar[13], axis=-1)
    v71 = tf.reduce_sum(wpar[14], axis=-1)
    v81 = tf.reduce_sum(wpar[15], axis=-1)

    f2 = (gamma * (v11+v21+v31+v41+v51+v61+v71+v81))
    f3 = (lada * (v1+v2+v3+v4+v5+v6+v7+v8))
    cost_general = f2+f3

    m = [tf.nn.l2_loss(tf.cast(v, tf.float32))for v in tf.trainable_variables() if 'bias' not in v.name]

    l2_loss = regular_fac * tf.add_n(m)
    cost_total = (cost_general+l2_loss)/batch_size
    return cost_total,f2,f3,l2_loss,cost_general


# def cost (y_pred, y_true):
#
#     y = tf.reshape(y_true, [-1, 26])
#     y_hat = tf.reshape(y_pred, [-1, 8])
#
#     v1 = ((-1) * tf.sign(y[:, 2]) + 1) * (tf.squared_difference(y_hat[:, 0], tf.sign(y[:, 2])))
#     v2 = ((-1) * tf.sign(y[:, 5]) + 1) * (tf.squared_difference(y_hat[:, 1], tf.sign(y[:, 5])))
#     v3 = ((-1) * tf.sign(y[:, 8]) + 1) * (tf.squared_difference(y_hat[:, 2], tf.sign(y[:, 8])))
#     v4 = ((-1) * tf.sign(y[:, 11]) + 1) * (tf.squared_difference(y_hat[:, 3], tf.sign(y[:, 11])))
#     v5 = ((-1) * tf.sign(y[:, 14]) + 1) * (tf.squared_difference(y_hat[:, 4], tf.sign(y[:, 14])))
#     v6 = ((-1) * tf.sign(y[:, 17]) + 1) * (tf.squared_difference(y_hat[:, 5], tf.sign(y[:, 17])))
#     v7 = ((-1) * tf.sign(y[:, 20]) + 1) * (tf.squared_difference(y_hat[:, 6], tf.sign(y[:, 20])))
#     v8 = ((-1) * tf.sign(y[:, 21]) + 1) * (tf.squared_difference(y_hat[:, 7], tf.sign(y[:, 21])))
#
#     v11 = tf.sign(y[:, 2]) * (tf.squared_difference(y_hat[:, 0], tf.sign(y[:, 2])))
#     v21 = tf.sign(y[:, 5]) * (tf.squared_difference(y_hat[:, 1], tf.sign(y[:, 5])))
#     v31 = tf.sign(y[:, 8]) * (tf.squared_difference(y_hat[:, 2], tf.sign(y[:, 8])))
#     v41 = tf.sign(y[:, 11]) * (tf.squared_difference(y_hat[:, 3], tf.sign(y[:, 11])))
#     v51 = tf.sign(y[:, 14]) * (tf.squared_difference(y_hat[:, 4], tf.sign(y[:, 14])))
#     v61 = tf.sign(y[:, 17]) * (tf.squared_difference(y_hat[:, 5], tf.sign(y[:, 17])))
#     v71 = tf.sign(y[:, 20]) * (tf.squared_difference(y_hat[:, 6], tf.sign(y[:, 20])))
#     v81 = tf.sign(y[:, 21]) * (tf.squared_difference(y_hat[:, 7], tf.sign(y[:, 21])))
#
#     cost = (v1,v2,v3,v4,v5,v6,v7,v8,v11,v21,v31,v41,v51,v61,v71,v81)
#     return cost

def cost (y_pred, y_true):

    y = tf.reshape(y_true, [-1, 26])
    y_hat = tf.reshape(y_pred, [-1, 8])

    v1 = (-1) * ((-1) * tf.sign(y[:, 2]) + 1) * (tf.log(1 - y_hat[:, 0]))
    v2 = (-1) * ((-1) * tf.sign(y[:, 5]) + 1) * (tf.log(1 - y_hat[:, 1]))
    v3 = (-1) * ((-1) * tf.sign(y[:, 8]) + 1) * (tf.log(1 - y_hat[:, 2]))
    v4 = (-1) * ((-1) * tf.sign(y[:, 11]) + 1) * (tf.log(1 - y_hat[:, 3]))
    v5 = (-1) * ((-1) * tf.sign(y[:, 14]) + 1) * (tf.log(1 - y_hat[:, 4]))
    v6 = (-1) * ((-1) * tf.sign(y[:, 17]) + 1) * (tf.log(1 - y_hat[:, 5]))
    v7 = (-1) * ((-1) * tf.sign(y[:, 20]) + 1) * (tf.log(1 - y_hat[:, 6]))
    v8 = (-1) * ((-1) * tf.sign(y[:, 21]) + 1) * (tf.log(1 - y_hat[:, 7]))

    v11 = (-1) * tf.sign(y[:, 2]) * (tf.log(y_hat[:, 0]))  # in model_1280relu_7 series there is no sign() for the
    v21 = (-1) * tf.sign(y[:, 5]) * (tf.log(y_hat[:, 1]))  # y_hat in the error calculation just one for before the error
    v31 = (-1) * tf.sign(y[:, 8]) * (tf.log(y_hat[:, 2]))
    v41 = (-1) * tf.sign(y[:, 11]) * (tf.log(y_hat[:, 3]))
    v51 = (-1) * tf.sign(y[:, 14]) * (tf.log(y_hat[:, 4]))
    v61 = (-1) * tf.sign(y[:, 17]) * (tf.log(y_hat[:, 5]))
    v71 = (-1) * tf.sign(y[:, 20]) * (tf.log(y_hat[:, 6]))
    v81 = (-1) * tf.sign(y[:, 21]) * (tf.log(y_hat[:, 7]))

    cost = (v1,v2,v3,v4,v5,v6,v7,v8,v11,v21,v31,v41,v51,v61,v71,v81)
    return cost