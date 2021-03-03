import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D

def correlationLayer():
    
    return None

def net():
    input_l = Input((None, None, 3))
    input_r = Input((None, None, 3))

    conv1_l = Conv2D(64,(7,7), padding='same')(input_l)
    conv1_l = MaxPooling2D((2,2))(conv1_l)
    conv1_l = ReLU()(conv1_l)
    conv1_r = Conv2D(64,(7,7), padding='same')(input_r)
    conv1_r = MaxPooling2D((2,2))(conv1_r)
    conv1_r = ReLU()(conv1_r)

    conv2_l = Conv2D(128,(5,5), padding='same')(conv1_l)
    conv2_l = MaxPooling2D((2,2))(conv2_l)
    conv2_l = ReLU()(conv2_l)
    conv2_r = Conv2D(128,(5,5), padding='same')(conv1_r)
    conv2_r = MaxPooling2D((2,2))(conv2_r)
    conv2_r = ReLU()(conv2_r)

    conv3_l = Conv2D(256,(5,5), padding='same')(conv2_l)
    conv3_l = MaxPooling2D((2,2))(conv3_l)
    conv3_l = ReLU()(conv3_l)
    conv3_r = Conv2D(256,(5,5), padding='same')(conv2_r)
    conv3_r = MaxPooling2D((2,2))(conv3_r)
    conv3_r = ReLU()(conv3_r)
    

