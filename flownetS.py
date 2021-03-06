import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import sys
import cv2
import copy
import warnings
import numpy as np
import time
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from skimage.io import imread, imshow, show
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dropout, Lambda
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, BatchNormalization
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import ReLU
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler

from utilities.utils import utilsProcessing
from utilities.utils import preprocessing

def epe_loss_function(y_actual, y_predicted):
    return tf.reduce_mean(tf.norm(y_actual - y_predicted, ord = 2, axis = -1))

def scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * tf.math.exp(-1.0)

class flownetS:
    def __init__(self):
        self.IMG_CHANNELS = 3

    def net(self):
        inputs = Input((None, None, self.IMG_CHANNELS*2))

        inputs_norm = Lambda(lambda x:x / 255)(inputs)

        conv1 = Conv2D(64,(7,7), padding='same')(inputs_norm)
        conv1 = MaxPooling2D((2,2))(conv1)
        conv1 = BatchNormalization()(conv1)
        conv1 = ReLU()(conv1)

        conv2 = Conv2D(128,(5,5), padding='same')(conv1)
        conv2 = MaxPooling2D((2,2))(conv2)
        conv2 = BatchNormalization()(conv2)
        conv2 = ReLU()(conv2)

        conv3 = Conv2D(256,(3,3), padding='same')(conv2)
        conv3 = MaxPooling2D((2,2))(conv3)
        conv3 = BatchNormalization()(conv3)
        conv3 = ReLU()(conv3)

        conv3_1 = Conv2D(256,(3,3), padding='same')(conv3)
        conv3_1 = BatchNormalization()(conv3_1)
        conv3_1 = ReLU()(conv3_1)

        conv4 = Conv2D(512,(3,3), padding='same')(conv3_1)
        conv4 = MaxPooling2D((2,2))(conv4)
        conv4 = BatchNormalization()(conv4)
        conv4 = ReLU()(conv4)

        conv4_1 = Conv2D(512,(3,3), padding='same')(conv4)
        conv4_1 = BatchNormalization()(conv4_1)
        conv4_1 = ReLU()(conv4_1)

        conv5 = Conv2D(512,(3,3), padding='same')(conv4_1)
        conv5 = MaxPooling2D((2,2))(conv5)
        conv5 = BatchNormalization()(conv5)
        conv5 = ReLU()(conv5)

        conv5_1 = Conv2D(512,(3,3), padding='same')(conv5)
        conv5_1 = BatchNormalization()(conv5_1)
        conv5_1 = ReLU()(conv5_1)

        conv6 = Conv2D(1024,(3,3), padding='same')(conv5_1)
        conv6 = MaxPooling2D((2,2))(conv6)
        conv6 = BatchNormalization()(conv6)
        conv6 = ReLU()(conv6)

        deconv5 = Conv2DTranspose(512,(1,1), strides=(2,2), padding='same')(conv6)
        deconv5 = BatchNormalization()(deconv5)
        deconv5 = Concatenate()([deconv5,conv5_1])
        flow5 = Conv2DTranspose(2,(5,5),strides=(2,2), padding='same')(deconv5)

        deconv4 = Conv2DTranspose(256,(1,1), strides=(2,2), padding='same')(deconv5)
        deconv4 = BatchNormalization()(deconv4)
        deconv4 = Concatenate()([deconv4,conv4_1,flow5])
        flow4 = Conv2DTranspose(2,(5,5),strides=(2,2), padding='same')(deconv4)

        deconv3 = Conv2DTranspose(128,(1,1), strides=(2,2), padding='same')(deconv4)
        deconv3 = BatchNormalization()(deconv3)
        deconv3 = Concatenate()([deconv3,conv3_1,flow4])
        flow3 = Conv2DTranspose(2,(5,5),strides=(2,2), padding='same')(deconv3)

        deconv2 = Conv2DTranspose(64,(1,1), strides=(2,2), padding='same')(deconv3)
        deconv2 = BatchNormalization()(deconv2)
        deconv2 = Concatenate()([deconv2,conv2,flow3])

        outputs = Conv2DTranspose(2,(5,5),strides=(4,4), padding='same')(deconv2)

        model = Model(inputs=[inputs], outputs=[outputs])
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])	
        return model
	
    def displayResults():
        plt.plot(results.history['accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train'], loc='upper left')
        plt.show()

        plt.plot(results.history['accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train'], loc='upper left')
        plt.show()
		
    def makePrediction(img1,img2):
        model = load_model('./models/flowNetSimple_500.h5')
        img_stack = np.zeros((1, img1.shape[0], img1.shape[1], IMG_CHANNELS*2), dtype = np.uint8)
        img_stack[0] = np.concatenate((img1, img2), axis = 2)
        pred_test = model.predict(img_stack, verbose = 1)

		
if __name__ == '__main__':
    
    cnt_n = 0

    obj = flownetS()
    
    model = obj.net()
    model.summary()

    objpre = preprocessing()
    x, y = objpre.setPathSintel()
    x_f, y_f = objpre.listFiles(x,y)
    x_files, y_files = objpre.listFileAsDict(x_f,y_f,x,y)

    for idx, key in enumerate(x_files.keys()):
        cnt_n += len(x_files[key])


    X_train = np.zeros(((cnt_n-(len(x_files.keys())*2))*3, objpre.IMG_HEIGHT, objpre.IMG_WIDTH, objpre.IMG_CHANNELS*2), dtype=np.uint8)
    Y_train = np.zeros(((cnt_n-(len(x_files.keys())*2))*3, objpre.IMG_HEIGHT, objpre.IMG_WIDTH, objpre.FLO_CHANNELS), dtype=np.float32)

    obj.createDataset(X_train, Y_train, x, y, x_files, y_files)

    #checkpointer = ModelCheckpoint('./models/flownetS-weights-500.ckpt', verbose=1, save_weights_only=True, save_best_only=True)
    #keras_checkpointers = [ModelCheckpoint('./models/flownetS-weights-500.ckpt', verbose=1, save_weights_only=True, save_best_only=True), LearningRateScheduler(scheduler)]

    checkpointer = ModelCheckpoint('./models/flowNetS-complete-500.h5', verbose=1, save_best_only=True)
    results = model.fit(X_train, Y_train, validation_split=0.1, batch_size=32, shuffle=True, epochs=500, callbacks=[checkpointer])

    hist_df = pd.DataFrame(results.history)
    hist_csv_file = './models/flownetS-history-500.csv'
    
    with open(hist_csv_file, mode = 'w') as f:
        hist_df.to_csv(f)
    
    print('End of Training!')
