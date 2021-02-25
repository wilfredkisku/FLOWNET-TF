'''
This program is free software: you can use, modify and/or redistribute it
under the terms of the simplified BSD License. You should have received a
copy of this license along this program.

Copyright 2020, wilfred kisku <kisku.1@iitj.ac.in>
All rights reserved.
'''

import os
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
import sys
from imageio import imread

import pathlib as Path
#import tensorflow as tf

class preprocessing:
    def setPathSintel():
        '''
        Exrtact the 
        '''

        TRAIN_PATH = '/workspace/storage/flownet-tf/data/training/'
        TEST_PATH = '/workspace/storage/flownet-tf/data/test/'

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
        cnt_iter = 0
        #iterates over the number of directories in the training folder (needs to iterate over all the folders)
        for i in range(len(x_dirs)):
            #iterate over the directories that contain a particular scenes
            for j in tqdm(range(len(list_x_dirs)), desc="Processing"):
                #iterate over files inside each of these folders that contain the frame t,t+1, .... , t+n instances
                for k in range(len(x_files_dict[list_x_dirs[j]])-2):

                    img_x_1_ = cv2.imread(str(p.joinpath(x_dirs[i]).joinpath(list_x_dirs[j]).joinpath(x_files_dict[list_x_dirs[j]][k])), cv2.IMREAD_COLOR)
                    img_x_2_ = cv2.imread(str(p.joinpath(x_dirs[i]).joinpath(list_x_dirs[j]).joinpath(x_files_dict[list_x_dirs[j]][k+1])), cv2.IMREAD_COLOR)

                    img_x_1 = img_x_1_[26:410,l:r,:]
                    img_x_2 = img_x_2_[26:410,l:r,:]

                    img_stacked = np.concatenate((img_x_1 ,img_x_2),axis=2)
                    X_train[cnt_iter]= img_stacked

                    #method call to the flow to vector conversion algorithmin the utilities
                    img_y_ = utilsProcessing.flowToArray(str(p.joinpath(y_dirs[0]).joinpath(list_y_dirs[j]).joinpath(y_files_dict[list_y_dirs[j]][k+1])))
                    Y_train[cnt_iter]= img_y_[26:410,l:r,:]

                    cnt_iter += 1

        return X_train, Y_train


'''
className: utilsProcessing
methods: quiverPlot()
	 filesDisplay()
	 flowToArray()
'''
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
        '''
        display the sequence of images in the path diretory provided 
        at a frame rate of 1FPS. 
        '''
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

    def quiverPlot(flow):
        '''
        Plot the quiver for the horizontal and vertical
        '''
        steps = 20
        plt.quiver(np.arange(0,flow.shape[1],steps), np.arange(flow.shape[0], -1, -steps), flow[::steps, ::steps, 0], flow[::steps, ::steps, 1])
        plt.savefig("flowimage")
        return None

    def flowToArray(flowfile):
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
    Confined testing of the module.
    '''
    #utilsProcessing.quiverPlot()
    #utilsProcessing.filesDisplay()
