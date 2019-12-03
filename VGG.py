from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
import numpy as np
from tensorflow.keras.applications.Mobile

def VGG_16():
    model = VGG19(weights='imagenet', include_top=True)
    return model




