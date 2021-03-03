import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D

def net():
    input_l = Input((None, None, 3))
    input_r = Input((None, None, 3))

    conv1 = Conv2D()(input_l)
    conv1_l = Conv2D()(input_r)
