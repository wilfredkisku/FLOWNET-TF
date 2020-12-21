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
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from utilities.utils import utilsProcessing

#images reduced to these parameters
IMG_HEIGHT = 384
IMG_WIDTH = 512
IMG_CHANNELS = 3
FLO_CHANNELS = 2

x_dirs = ['albedo', 'clean', 'final']
y_dirs = ['flow']

l = 256
r = 768

'''
Class Name: preprocessing
methods: setPathSintel()
         listFiles()
         listFilesAsDict()
         createDataset()
'''

class preprocessing:
    def setPathSintel():
        '''
        Exrtact the 
        '''
        
        TRAIN_PATH = 'data/training/'
        TEST_PATH = 'data/test/'

        p = Path(TRAIN_PATH)
        
        list_x_dirs = []
        list_y_dirs = []

        try:
            list_x_dirs = next(os.walk(p.joinpath(x_dirs[0])))[1]
            list_y_dirs = next(os.walk(p.joinpath(y_dirs[0])))[1]
        except StopIteration:
            pass

        list_x_dirs.sort()
        list_y_dirs.sort()
        return p, list_x_dirs, list_y_dirs

    def listFiles(p, list_x_dirs, list_y_dirs):
        x_files_dict = {dirs_: [] for dirs_ in list_x_dirs}
        y_files_dict = {dirs_: [] for dirs_ in list_y_dirs}
        #lst = os.listdir(p.joinpath(x_dirs[0]).joinpath(list_x_dirs[0]))
        #lst.sort()
        return x_files_dict, y_files_dict

    def listFileAsDict(p, x_files_dict, y_files_dict, list_x_dirs, list_y_dirs):
        for i in range(len(list_x_dirs)):
            n_lst = os.listdir(p.joinpath(x_dirs[0]).joinpath(list_x_dirs[i]))
            n_lst.sort()
            x_files_dict[list_x_dirs[i]] = n_lst

        for i in range(len(list_y_dirs)):
            n_lst = os.listdir(p.joinpath(y_dirs[0]).joinpath(list_y_dirs[i]))
            n_lst.sort()
            y_files_dict[list_y_dirs[i]] = n_lst
        return x_files_dict, y_files_dict

    def createDataset(X_train, Y_train, cnt_iter):

        for i in range(len(x_dirs)):
            for j in tqdm(range(len(list_x_dirs)), desc="Processing"):
                for k in range(len(x_files_dict[list_x_dirs[j]])-1):
                    img_x_1 = cv2.imread(str(p.joinpath(x_dirs[i]).joinpath(list_x_dirs[j]).joinpath(x_files_dict[list_x_dirs[j]][k])), cv2.IMREAD_COLOR)
                    img_x_2 = cv2.imread(str(p.joinpath(x_dirs[i]).joinpath(list_x_dirs[j]).joinpath(x_files_dict[list_x_dirs[j]][k+1])), cv2.IMREAD_COLOR)
                    img_x_1 = img_x_1[26:410,l:r,:]
                    img_x_2 = img_x_2[26:410,l:r,:]
                    img_stacked = np.concatenate((img_x_1 ,img_x_2),axis=2)
                    X_train[cnt_iter] = img_stacked

                    #method call to the flow to vector conversion algorithmin the utilities
                    #img_y_ = flo_to_vector(str(p.joinpath(y_dirs[0]).joinpath(list_y_dirs[j]).joinpath(y_files_dict[list_y_dirs[j]][k])))
                    img_y_ = flowToArray(str(p.joinpath(y_dirs[0]).joinpath(list_y_dirs[j]).joinpath(y_files_dict[list_y_dirs[j]][k])))
                    Y_train[cnt_iter] = img_y_[26:410,l:r,:]
			
                    cnt_iter += 1
        return X_train, Y_train

    def createDatasetPrediction(X_train, Y_train, list_x_dirs, list_y_dirs, x_files_dict, y_files_dict):
        #print(X_train.shape)
        #print(Y_train.shape)

        #print(list_x_dirs)
        #print(list_y_dirs)

        #print(x_files_dict)
        #print(y_files_dict)
        
        cnt_iter = 0
        #iterates over the number of directories in the training folder (needs to iterate over all the folders)
        for i in range(len(x_dirs)):
            #iterate over the directories that contain a particular scenes
            for j in tqdm(range(len(list_x_dirs)), desc="Processing"):
                #iterate over files inside each of these folders that contain the frame t,t+1, .... , t+n instances
                for k in range(len(x_files_dict[list_x_dirs[j]])-2):
        
                    #img_x_1_ = cv2.imread(str(p.joinpath(x_dirs[i]).joinpath(list_x_dirs[j]).joinpath(x_files_dict[list_x_dirs[j]][k])), cv2.IMREAD_COLOR)
                    #img_x_2_ = cv2.imread(str(p.joinpath(x_dirs[i]).joinpath(list_x_dirs[j]).joinpath(x_files_dict[list_x_dirs[j]][k+1])), cv2.IMREAD_COLOR)
                    
                    #img_x_1 = img_x_1_[26:410,l:r,:]
                    #img_x_2 = img_x_2_[26:410,l:r,:]
                    
                    #img_stacked = np.concatenate((img_x_1 ,img_x_2),axis=2)
                    #X_train[cnt_iter] = img_stacked

                    #method call to the flow to vector conversion algorithmin the utilities
                    img_y_ = utilsProcessing.flowToArray(str(p.joinpath(y_dirs[0]).joinpath(list_y_dirs[j]).joinpath(y_files_dict[list_y_dirs[j]][k+1])))
                    Y_train[cnt_iter] = img_y_[26:410,l:r,:]

                    cnt_iter += 1
        
        return X_train, Y_train
        

'''
Class Name: flownetS
methods: custom_loss_function()
         net()
         displayResults()
         makePrediction()
         mainFlowS()
'''
class flownetS:

    def custom_loss_function(y_actual, y_predicted):
        '''
        compute the loss function that amounts to the mean squared error. 
        '''
        y_predicted = tf.convert_to_tensor(y_predicted)
        y_actual = tf.cast(y_actual, y_predicted.dtype)
        custom_loss_value = tf.math.reduce_mean(tf.math.reduce_sum(tf.norm(y_actual - y_predicted)))
        return custom_loss_value

    def net():
        '''	
        Method defines the input/output and the network architecture model.
        '''
        inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS*2))

        #since the images are normalized already in the 
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
        '''
        Plot the Accuracy vs Epochs and Loss vs Epochs.
        '''
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
        return None
		
    def makePrediction(img1,img2):
        '''
        Make prediction on the saved model in the ./models folder.
        The file dimensions are to be 384 X 512 X 3 and after the two images are concatenated to form the input
        it is supposed to be of the dimension 384 X 512 X 6.
        '''
        model = load_model('./models/flowNetSimple_1000.h5')
        img_stack = np.zeros((1, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS*2), dtype = np.uint8)
        img_stack[0] = np.concatenate((img1, img2), axis = 2)
        pred_test = model.predict(img_stack, verbose = 1)

        #plot the quiver for the pred_test
        print(pred_test[0].shape)
        print(pred_test[0])
        return None

    def mainFlownetS():
        '''
        Main flownet simple method to call the other methods and test the general algorithm.
        Preprocessing would require a lot of data handle and cleaning and also tensorflow pipelining of data.
        '''
        X_train = np.zeros((cnt_n*3,IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS*2), dtype=np.uint8)
        Y_train = np.zeros((cnt_n*3,IMG_HEIGHT, IMG_WIDTH, FLO_CHANNELS), dtype=np.float32)
		
        #dataset preprocessing
        sys.stdout.flush()
        print('Getting and stacking train images and label images ... ')
        cnt_iter = 0
        for i in range(1):
            l = 256
            r = 768
            X_train, Y_train = preprocessing.crateDataset(X_train, Y_train, cnt_iter, l, r)
        sys.stdout.flush()

        #model creation and training/save
        model = flownetS.net()
        model.summary()
        checkpointer = ModelCheckpoint('models/flowNetSimple_1000.h5', verbose=1, save_best_only=True)
        results = model.fit(X_train, Y_train, validation_split=0.1, batch_size=32, shuffle=True, epochs=500, callbacks=[checkpointer])

        hist_df = pd.DataFrame(results.history)
        hist_csv_file = 'models/history.csv'
        with open(hist_csv_file, mode='w') as f:
            hist_df.to_csv(f)
		
        #prediction and validation of the trained model

if __name__ == '__main__':
    '''
    Test the methods for actual deploy.
    '''
    
    ##### Phase I #####
    p, lst_x_dirs, lst_y_dirs = preprocessing.setPathSintel()
    x_files_dict, y_files_dict = preprocessing.listFiles(p, lst_x_dirs, lst_y_dirs)

    x_files_dict, y_files_dict = preprocessing.listFileAsDict(p, x_files_dict, y_files_dict, lst_x_dirs, lst_y_dirs)

    #print(x_files_dict)
    #print(y_files_dict)

    cnt_x = 0
    cnt_y = 0
    for i in x_files_dict:
        if isinstance(x_files_dict[i], list):
            cnt_x += len(x_files_dict[i])
    
    for j in y_files_dict:
        if isinstance(y_files_dict[j], list):
            cnt_y += len(y_files_dict[j])

    #print('The count of the number of file in the x folder: ', cnt_x)
    #print('The count of the number of file in the y folder: ', cnt_y)
    
    ##### Phase II #####
    '''
    The preprocessed dataset need to be assigned a format that incuded cosecutive frames as the input.
    Also, the y values are the [U,V] vectors that would be extracted from the training.

    '''
    cnt_n = cnt_y - 23

    X_train = np.zeros((cnt_n*3,IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS*2), dtype=np.uint8)
    Y_train = np.zeros((cnt_n*3,IMG_HEIGHT, IMG_WIDTH, FLO_CHANNELS), dtype=np.float32)

    #print(X_train.shape)
    #print(Y_train.shape)

    X_train, Y_train = preprocessing.createDatasetPrediction(X_train, Y_train, lst_x_dirs, lst_y_dirs, x_files_dict, y_files_dict)
    
    print(Y_train[cnt_n*3 - 1])
    ##### Phase III #####
    '''
    model = flownetS.net()
    model.summary()
    checkpointer = ModelCheckpoint('models/flowNetSimple_1000.h5', verbose=1, save_best_only=True)
    results = model.fit(X_train, Y_train, validation_split=0.1, batch_size=32, shuffle=True, epochs=500, callbacks=[checkpointer])

    hist_df = pd.DataFrame(results.history)
    hist_csv_file = 'models/history.csv'
    '''
    ##### Modules test #####
    #flow = utilsProcessing.flowToArray('/home/wilfred/Datasets/MPI-Sintel-complete/training/flow/alley_1/frame_0001.flo')
    #print("The horizontal direction vector.")
    #print(flow[0])
    #print(flow[0].max(), flow[0].min())
    #print("The vertical direction vector.")
    #print(flow[1])
    #print(flow[1].max(), flow[1].min())
    #utilsProcessing.quiverPlot(flow)
