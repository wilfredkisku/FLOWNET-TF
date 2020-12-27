'''
This program is free software: you can use, modify and/or redistribute it
under the terms of the simplified BSD License. You should have received a
copy of this license along this program.
Copyright 2020, wilfred kisku <kisku.1@iitj.ac.in>
All rights reserved.
'''
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import pandas as pd
import numpy as np
import cv2 
import math
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from scipy.ndimage.filters import convolve as filter2

from utilities.classicalUtils import classicalUtilitiesHS as cuhs
from utilities.classicalUtils import classicalUtilitiesPY as cupy
from utilities.cameraUtils import Camera

from tensorflow.keras.models import load_model

#testing classicalUtils
def testingUtils():
    img1 = cv2.imread('./data/foreman/frame5.png', cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread('./data/foreman/frame7.png', cv2.IMREAD_GRAYSCALE)

    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)

    img1, img2 = cupy.normalize(img1,img2)

    img1 = cupy.gaussianSmoothing(img1)
    img2 = cupy.gaussianSmoothing(img2)

    cupy.scaling(img1)

    #testing utils
    taxis_frames = list(Path('./data/taxi').iterdir())
    #for i in taxis_frames:
    #	print(i)
    taxi1 = Image.open('./data/foreman/frame1.png')
    taxi2 = Image.open('./data/foreman/frame3.png')

    taxi33 = Image.open('./data/foreman/frame33.png')
    taxi35 = Image.open('./data/foreman/frame35.png')

    flow = cv2.calcOpticalFlowFarneback(np.array(taxi1), np.array(taxi2) , None, 0.5, 3, 15, 3, 5, 1.2, 0)
    flow_ = cv2.calcOpticalFlowFarneback(np.array(taxi33), np.array(taxi35) , None, 0.5, 3, 15, 3, 5, 1.2, 0)
    step = 5

    plt.quiver(np.arange(0, flow.shape[1], step), np.arange(flow.shape[0], -1, -step), flow[::step, ::step, 0], flow[::step, ::step, 1])
    plt.quiver(np.arange(0, flow_.shape[1], step), np.arange(flow_.shape[0], -1, -step), flow_[::step, ::step, 0], flow_[::step, ::step, 1])

    plt.savefig('taxis1-2')
    plt.savefig('taxis33-35')
    
#testing camera
def testingCamera():
    Camera.cameraMod()

#testing models
def testingModels():
    model = load_model('models/flowNetSimple_2000.h5')
    history = pd.read_csv('models/history.csv')

    history_ = history.rename({'Unnamed: 0': 'seq'}, axis='columns')
    #print(history_['seq'])
    #print(history_.head())

    #Get current axis
    ax = plt.gca()
    ax.set_ylim(0,350)

    # line plot for math marks
    history_.plot(kind = 'line',
        x = 'seq',
        y = 'loss',
        color = 'green',ax = ax)

    # line plot for physics marks
    history_.plot(kind = 'line',
        x = 'seq',
        y = 'val_loss',
        color = 'blue',ax = ax)

    # set the title
    plt.title('Loss Plot')

    # show the plot
    plt.show()

    ax_ = plt.gca()
    # line plot for math marks
    history_.plot(kind = 'line',
        x = 'seq',
        y = 'accuracy',
        color = 'green',ax = ax_)

    # line plot for physics marks
    history_.plot(kind = 'line',
        x = 'seq',
        y = 'val_accuracy',
        color = 'blue',ax = ax_)

    # set the title
    plt.title('Accuracy Plot')

    # show the plot
    plt.show()

if __name__ == "__main__":
    testingModels()
