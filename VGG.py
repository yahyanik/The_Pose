from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
import numpy as np

def VGG_16():
    model = VGG16(weights='imagenet', include_top=True)
    return model
