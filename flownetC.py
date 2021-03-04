import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Lambda

def correlate():

    return Lambda(lamdba x: tf.reduce_sum(tf.multiply(x[0],x[1]), axis = -1, keep_dims=True), name='correlate')

def correlationLayer(max_disp=20,stride=2):
    for i in range(-max_disp, max_disp + stride, stride):
        for j in range(-max_disp, max_disp + stride, stride):
            slice_ = get_padded_stride(conv3_pool_r,i,j,height_8,width_8)
            current_layer = correlate([conv3_pool_l,slice_])
            layer_list.append(current_layer)

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
    

