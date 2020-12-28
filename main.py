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

from scipy import interpolate
from pathlib import Path
from PIL import Image
from scipy.ndimage.filters import convolve as filter2

from utilities.classicalUtils import classicalUtilitiesHS as cuhs
from utilities.classicalUtils import classicalUtilitiesPY as cupy
from utilities.cameraUtils import Camera

from tensorflow.keras.models import load_model

IMG_HEIGHT = 384
IMG_WIDTH = 512
IMG_CHANNELS = 3

#testing classicalUtils
def testingUtils():
    img1 = cv2.imread('data/foreman/frame5.png', cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread('data/foreman/frame7.png', cv2.IMREAD_GRAYSCALE)

    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)

    img1, img2 = cupy.normalize(img1,img2)

    img1 = cupy.gaussianSmoothing(img1)
    img2 = cupy.gaussianSmoothing(img2)

    cupy.scaling(img1)

    #testing utils
    taxis_frames = list(Path('data/taxi').iterdir())
    #for i in taxis_frames:
    #	print(i)
    taxi1 = Image.open('data/foreman/frame1.png')
    taxi2 = Image.open('data/foreman/frame3.png')

    taxi33 = Image.open('data/foreman/frame33.png')
    taxi35 = Image.open('data/foreman/frame35.png')

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

    files = os.listdir('data/foreman/')
    files.sort()

    test_files = np.zeros((len(files)-2,IMG_HEIGHT,IMG_WIDTH,IMG_CHANNELS*2), dtype=np.uint8)
    p_ = Path('data/foreman/')
    for f in range(0,len(files)-1):
        f1 = str(p_.joinpath(files[f]))
        f2 = str(p_.joinpath(files[f+1]))
        print(f1)
        img_a = cv2.imread(f1)
        img_b = cv2.imread(f2)
        f_a_np_r = cv2.resize(img_a, (IMG_WIDTH, IMG_HEIGHT), interpolation = cv2.INTER_CUBIC)
        f_b_np_r = cv2.resize(img_b, (IMG_WIDTH, IMG_HEIGHT), interpolation = cv2.INTER_CUBIC)

        f_stack = np.concatenate((f_a_np_r,f_b_np_r), axis = 2)
        test_files[f-1] = f_stack
    
    pred_test = model.predict(test_files, verbose=1)

    step = 8

    fig1, ax1 = plt.subplots(figsize=(5,5))
    ax1.quiver(np.arange(0, pred_test[0].shape[1], step), np.arange(pred_test[0].shape[0], 0, -step), pred_test[0][::step, ::step, 0], pred_test[0][::step, ::step, 1], color='r')
    plt.show()

    return pred_test

def singleModelTest():
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

    img_a = cv2.imread('data/cars/seq01.png')
    img_b = cv2.imread('data/cars/seq02.png')
    f_a_np_r = cv2.resize(img_a, (IMG_WIDTH, IMG_HEIGHT), interpolation = cv2.INTER_CUBIC)
    f_b_np_r = cv2.resize(img_b, (IMG_WIDTH, IMG_HEIGHT), interpolation = cv2.INTER_CUBIC)

    print(f_a_np_r.shape)
    print(f_b_np_r.shape)

    f_stack = np.concatenate((f_a_np_r,f_b_np_r), axis = 2)
    f_stack_ = np.zeros((1,IMG_HEIGHT,IMG_WIDTH,IMG_CHANNELS*2), dtype = np.uint8)
    f_stack_[0] = f_stack
    print(f_stack_.shape)
    pred_test = model.predict(f_stack_, verbose=1)
    return pred_test

def warpImage(img, flow):
    
    image_height = img.shape[0]
    image_width = img.shape[1]
    
    flow_height = flow.shape[0]
    flow_width = flow.shape[1]
    
    n = image_height * image_width
    
    (iy, ix) = np.mgrid[0:image_height, 0:image_width]
    (fy, fx) = np.mgrid[0:flow_height, 0:flow_width]
    
    fx = fx.astype(np.float64)
    fy = fy.astype(np.float64)
    fx += flow[:,:,0]
    fy += flow[:,:,1]
    
    mask = np.logical_or(fx <0 , fx > flow_width)
    mask = np.logical_or(mask, fy < 0)
    mask = np.logical_or(mask, fy > flow_height)
    
    fx = np.minimum(np.maximum(fx, 0), flow_width)
    fy = np.minimum(np.maximum(fy, 0), flow_height)
    
    points = np.concatenate((ix.reshape(n,1), iy.reshape(n,1)), axis=1)
    xi = np.concatenate((fx.reshape(n, 1), fy.reshape(n,1)), axis=1)
    warp = np.zeros((image_height, image_width, img.shape[2]))
    
    for i in range(img.shape[2]):
        channel = img[:, :, i]
        values = channel.reshape(n, 1)
        new_channel = interpolate.griddata(points, values, xi, method='cubic')
        new_channel = np.reshape(new_channel, [flow_height, flow_width])
        new_channel[mask] = 1
        warp[:, :, i] = new_channel.astype(np.uint8)

    return warp.astype(np.uint8)

if __name__ == "__main__":
    '''
    #obtain the predicted flow for the future frame
    pred_test = testingModels()

    #obtain a single test image from the folder
    img_test = cv2.imread('data/foreman/frame03.png')
    img = cv2.resize(img_test, (IMG_WIDTH, IMG_HEIGHT), interpolation = cv2.INTER_CUBIC)
    img_warped = warpImage(img, pred_test[0])
    #print(img_warped)

    plt.imshow(img_warped)
    plt.show()
    '''
    pred_test = singleModelTest()

    img_test = cv2.imread('data/cars/seq02.png')
    img = cv2.resize(img_test, (IMG_WIDTH, IMG_HEIGHT), interpolation = cv2.INTER_CUBIC)
    img_warped = warpImage(img, pred_test[0])
    
    plt.imshow(img_warped)
    plt.show()

    img_warped = cv2.resize(img_warped, (256, 190), interpolation = cv2.INTER_CUBIC)
    cv2.imwrite('data/cars/seq03_pred.png', img_warped)
