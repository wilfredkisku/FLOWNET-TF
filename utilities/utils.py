import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import sys
import cv2
import copy
import time
import warnings
import numpy as np
import pandas as pd
import pathlib as Path
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from os.path import isfile, join
from imageio import imread
from tqdm import tqdm
from pathlib import Path
from skimage.io import imread, imshow, show

#*********************************************#
#******* Data Processing tensorflow **********#
#*********************************************#

class preprocessingTF:
    #input pipeline needs to be verified
    def __init__():
        self.IMG_HEIGHT = 436
        self.IMG_WIDTH = 1024
        self.IMG_CHANNELS = 3
        self.FLO_CHANNELS = 2

        self.BATCH_SIZE = 32
        self.BUFFER_SIZE = 100

        self.x_dirs = ['clean','albedo','final']
        self.y_dirs = ['flow']
        self.TRAIN_PATH = '/home/wilfred/Datasets/testFolder/data/training/'
        self.TEST_PATH = '/home/wilfred/Datasets/testFolder/data/test/'

        def normalize(input_image, real_image):
            input_image = (input_image / 255.)
            real_image = (real_image / 255.)
            return input_image, real_image

    def load(self):
        image = tf.io.read_file(image_file)
        image = tf.image.decode_jpeg(image)

        w = tf.shape(image)[1]

        w = w // 2  
        real_image = image[:, :w, :]
        input_image = image[:, w:, :]

        input_image = tf.cast(input_image, tf.float32)
        real_image = tf.cast(real_image, tf.float32)

        return input_image, real_image

    def load_image_train(self, image_file):
        input_image, real_image = load(image_file)
        input_image, real_image = normalize(input_image, real_image)
        return input_image, real_image

    def load_image_test(self, image_file):
        input_image, real_image = load(image_file)
        input_image, real_image = normalize(input_image, real_image)
        return input_image, real_image

    def trainLoadMain(self):
        train_dataset = tf.data.Dataset.list_files(self.PATH)
        train_dataset = train_dataset.map(load_image_train, num_parallel_calls=tf.data.AUTOTUNE)
        train_dataset = train_dataset.shuffle(self.BUFFER_SIZE)
        train_dataset = train_dataset.batch(self.BATCH_SIZE)
        return None

    def testLoadMain(self):
        test_dataset = tf.data.Dataset.list_files(self.PATH)
        test_dataset = test_dataset.map(load_image_test)
        test_dataset = test_dataset.batch(self.BATCH_SIZE)
        return None

#****************************************#
#************* Data Processing **********#
#****************************************#

class preprocessing:
    def __init__(self):
        self.IMG_HEIGHT = 436
        self.IMG_WIDTH = 1024
        self.IMG_CHANNELS = 3
        self.FLO_CHANNELS = 2

        self.x_dirs = ['clean','albedo','final']
        self.y_dirs = ['flow']

        #self.TRAIN_PATH = '/home/wilfred/Datasets/testFolder/data/training/'
        #self.TEST_PATH = '/home/wilfred/Datasets/testFolder/data/test/'
        
        self.TRAIN_PATH = '../data/training/'
        self.TEST_PATH = '../data/test/'

    def setPathSintel(self):
        p = Path(self.TRAIN_PATH)
        list_x_dirs = []
        list_y_dirs = []
        print(os.listdir(p))

        try:
            list_x_dirs = next(os.walk(p.joinpath(self.x_dirs[0])))[1]
        except StopIteration:
            pass

        list_x_dirs.sort()
        list_y_dirs = copy.deepcopy(list_x_dirs)
        return list_x_dirs, list_y_dirs

    def listFiles(self, list_x_dirs, list_y_dirs):
        x_files_dict = {dirs_: [] for dirs_ in list_x_dirs}
        y_files_dict = {dirs_: [] for dirs_ in list_y_dirs}
        return x_files_dict, y_files_dict

    def listFileAsDict(self, x_files_dict, y_files_dict, list_x_dirs, list_y_dirs):
        p = Path(self.TRAIN_PATH)
        for i in range(len(list_x_dirs)):
            n_lst = os.listdir(p.joinpath(self.x_dirs[0]).joinpath(list_x_dirs[i]))
            n_lst.sort()
            x_files_dict[list_x_dirs[i]] = n_lst

        for i in range(len(list_y_dirs)):
            n_lst = os.listdir(p.joinpath(self.y_dirs[0]).joinpath(list_y_dirs[i]))
            n_lst.sort()
            y_files_dict[list_y_dirs[i]] = n_lst
        return x_files_dict, y_files_dict

    def createDataset(X_train, Y_train, list_x_dirs, list_y_dirs, x_files_dict, y_files_dict):
        
        p = Path(self.TRAIN_PATH)
        cnt_iter = 0
        
        for i in range(len(self.x_dirs)):
            for j in tqdm(range(len(list_x_dirs)), desc="Processing"):
                for k in range(len(x_files_dict[list_x_dirs[j]])-2):

                    img_x_1_ = cv2.imread(str(p.joinpath(self.x_dirs[i]).joinpath(list_x_dirs[j]).joinpath(x_files_dict[list_x_dirs[j]][k])), cv2.IMREAD_COLOR)
                    img_x_2_ = cv2.imread(str(p.joinpath(self.x_dirs[i]).joinpath(list_x_dirs[j]).joinpath(x_files_dict[list_x_dirs[j]][k+1])), cv2.IMREAD_COLOR)

                    img_stacked = np.concatenate((img_x_1 ,img_x_2),axis=2)
                    X_train[cnt_iter]= img_stacked

                    img_y_ = utilsProcessing.flowToArray(str(p.joinpath(y_dirs[0]).joinpath(list_y_dirs[j]).joinpath(y_files_dict[list_y_dirs[j]][k+1])))
                    Y_train[cnt_iter]= copy.deepcopy(img_y_)

                    cnt_iter += 1

        return X_train, Y_train

    def epeCalculate(self, actual, pred):
        #return tf.reduce_mean(tf.math.sqrt(tf.math.square(actual-pred)))
        return tf.reduce_mean(tf.norm(actual - pred, ord = 2, axis = -1))

    def predict(self, model_path, img_path_1, img_path_2):
        #requires the .h5 file that has the weights and baises from the trained model
        #requires the path for the two consecutive images for obtaining quiver plot
        
        model = load_model(model_path)
        img_a = cv2.imread(img_path_1)
        img_b = cv2.imread(img_path_2)
        img_a = img_a[:128,:,:]
        img_b = img_b[:128,:,:]
        plt.imshow(img_a)
        img_stack = np.concatenate((img_a,img_b), axis = 2)
        print(img_stack.shape)
        pred_flo = np.zeros((1,img_a.shape[0],img_a.shape[1],6), dtype = np.uint8)
        pred_flo[0] = img_stack
        pred_stack = model.predict(pred_flo, verbose=1)

        pred = pred_stack[0]
        step = 8
        plt.quiver(np.arange(0, pred.shape[1], step), np.arange(pred.shape[0], 0, -step), pred[::step, ::step, 0], pred[::step, ::step, 1], color='r')
        plt.savefig('/home/wilfred/Downloads/github/Python_Projects/flownet-tf/data/cars/predicted_seq_quiver.png')
        plt.show()
        return None


#*****************************************#
#*********** Flow Processing *************#
#*****************************************#

class utilsProcessing:
    def warpImage(curImg, flow):
        h, w = flow.shape[:2]
        flow = -flow
        flow[:,:,0] += np.arange(w)
        flow[:,:,1] += np.arange(h)[:,np.newaxis]
        prevImg = cv2.remap(curImg, flow, None, cv.INTER_LINEAR)

        plt.savefig()

        return None

    def filesDisplay(path):
        files = os.listdir(p)
        files.sort()
        for f in files:
            print(f)

        for i in files:
            img_ = cv2.imread(p+i,-1)
            cv2.imshow('Display', img_)
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()
        return None

    def quiverPlot(self, flow):
        steps = 32
        plt.quiver(np.arange(0,flow.shape[1],steps), np.arange(flow.shape[0], -1, -steps), flow[::steps, ::steps, 0], flow[::steps, ::steps, 1])
        plt.savefig("/home/wilfred/Downloads/github/Python_Projects/flownet-tf/data/flowimage.png")
        return None

    def flowToArray(self, flowfile):
        with open(flowfile, 'rb') as f:
            magic = np.fromfile(f, np.float32, count=1)
            if 202021.25 != magic:
                print('Magic number incorrect. Invalid .flo file!')
            else:
                w, h = np.fromfile(f, np.int32, count=2)
                data = np.fromfile(f, np.float32, count = 2* w * h)
                data2D = np.resize(data, (h, w, 2))
        return data2D

if __name__ == '__main__':

    '''
    img_path_1 = '/home/wilfred/Downloads/github/Python_Projects/flownet-tf/data/cars/seq01.png'
    img_path_2 = '/home/wilfred/Downloads/github/Python_Projects/flownet-tf/data/cars/seq02.png'
    path = '/home/wilfred/Downloads/flowNetS-complete-500.h5'
    
    obj = preprocessing()
    obj.predict(path,img_path_1,img_path_2)
    
    obj = utilsProcessing()
    flow = obj.flowToArray("/home/wilfred/Datasets/testFolder/data/training/flow/alley_1/frame_0001.flo")

    print(flow.shape)

    obj.quiverPlot(flow)
    
    '''
    obj = utilsProcessing()
    objpre = preprocessing()
    actual = obj.flowToArray("/home/wilfred/Datasets/testFolder/data/training/flow/alley_1/frame_0001.flo")
    pred = obj.flowToArray("/home/wilfred/Datasets/testFolder/data/training/flow/alley_1/frame_0002.flo")

    print(objpre.epeCalculate(actual, pred).numpy())
    
