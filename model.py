from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np
import tensorflow as tf

import cv2
import imutils

def Data ():

def model (features, labels, mode):
    """Model function for CNN."""
    # Input Layer

    input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])

