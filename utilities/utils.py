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
