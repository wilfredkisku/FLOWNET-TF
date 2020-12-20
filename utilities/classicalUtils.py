'''
This program is free software: you can use, modify and/or redistribute it
under the terms of the simplified BSD License. You should have received a
copy of this license along this program.

Copyright 2020, wilfred kisku <kisku.1@iitj.ac.in>
All rights reserved.
'''
from pathlib import Path
from scipy.ndimage.filters import convolve as filter2

import os
import cv2
import random
import numpy as np

#parameters
alpha_py = 10
nscales	= 6 #same as the number of warps
zfactor = 0.5
warps = 6 #same as the number of scales (during downsampling)
TOL = 0.0001 #stopping criterion threshold
maxiter =150

#filters 
wavg_ker = np.array([[1/12,2/12,1/12],[2/12,0,2/12],[1/12,2/12,1/12]])
xgra_ker = np.array([[-1,1],[-1,1]]) * 0.25
ygra_ker = np.array([[-1,-1],[1,1]]) * 0.25
t_ker = np.ones((2,2)) * 0.25
'''
className: classicalUtilitiesHS
methods: derivative()
	 stoppingCriterion()
	 costFunction()
	 displayQuiver() 
	 mainHSclassical()
'''
class classicalUtilitiesHS:
    '''
    Computes the derivatives Ex, Ey and Et where these are the intensities.
    '''
    def derivative(img1, img2):
        img1 = img1.astype(np.float32)
        img2 = img2.astype(np.float32)

        Ex = filter2(img1, xgra_ker) + filter2(img2, xgra_ker)
        Ey = filter2(img1, ygra_ker) + filter2(img2, ygra_ker)
        Et = filter2(img1, t_ker) + filter2(img2, -t_ker)
        return (Ex, Ey, Et)

    '''
    Computes the stopping criterion epsilon.
    '''
    def stoppingCriterion(u_new, v_new, u_old, v_old):
        epsilon_evaluated = np.mean((u_new - u_old)**2 + (v_new - v_old)**2)
        return epsilon_evaluated

    '''
    Computes the solution to the linear equation with the iterative method.
    The method needs to be called iteratively.
    '''
    def approximations(U, V, Ex, Ey, Et, alpha):

        Uavg = filter2(U, wavg_ker)
        Vavg = filter2(V, wavg_ker)

        tmp = (Ex*Uavg) + (Ey*Vavg) + Et
        tmp /= alpha**2 + Ex**2 + Ey**2

        U = Uavg - (Ex*tmp)
        V = Vavg - (Ey*tmp)

        return (U, V)

    #carries out the complete algorithm for classical HS method
    '''
    Compute the values if Ix, Iy and It.
    Intialize the values if u,v and n to 0.
    Iterate over Nmaxiter and stoppingCriterion (epsilon).
    compute the value of unew and vnew.
    change the values if u and v.
    compute stopping criterion.
    increment the iter until it reaches Nmaxiter.
    '''
    def mainHSclassical(img_path1, img_path2):
        img1 = cv2.imread(img_path1, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(img_path2, cv2.IMREAD_GRAYSCALE)

        img1 = img1.astype(np.float32)
        img2 = img2.astype(np.float32)
        return None

'''
ClassName: classicalUtilitiesPy
Methods: normalize()
	 gaussianSmoothing()
	 scaling()
	 computeOpticalFlow()
	 mainHSpyramidal()	
'''
class classicalUtilitiesPY:

    def normalize(img1_, img2_):
        img1_max, img1_min = img1_.max(), img1_.min()
        img2_max, img2_min = img2_.max(), img2_.min()
        img1_ = 255. * ((img1_ - img1_min) / (img1_max - img1_min))
        img2_ = 255. * ((img2_ - img2_min) / (img2_max - img2_min))

        return img1_, img2_

    def gaussianSmoothing(img_):
        g = cv2.getGaussianKernel(5, 0.8)
        g_t = np.transpose(g)
        g_ker = (255. * (g * g_t))

        g_ker_sum = sum(sum(g_ker))

        img_g = filter2(img_, (1/g_ker_sum) * g_ker)

        return img_g

    def scaling(img_):
        nx = []
        ny = []
        for i in range(nscales):
            print('The shapes are : ',img_.shape[0],' and ',img_.shape[1])
            nx.append([img_.shape[0]])
            ny.append([img_.shape[1]])
            img_ = cv2.resize(img_,(round(zfactor * img_.shape[1]), round(zfactor * img_.shape[0])), interpolation = cv2.INTER_CUBIC)
            #display the interpolated image
            cv2.imshow('interpolated and resized', img_.astype(np.uint8))
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            print(nx, ny)	
        return None

    def computeOpticalFlow():
        '''To Do'''
        return None

    def mainHSpyramidal(img1, img2):
        img1, img2 = classicalUtilitiesPY.normalize(img1, img2)
        img1 = classicalUtilitiesPY.gaussianSmoothing(img1)
        img2 = classicalUtilitiesPY.gaussianSmoothing(img2)

        return None

if __name__ == "__main__":

    img1 = cv2.imread('../data/foreman/frame1.png',0)
    img2 = cv2.imread('../data/foreman/frame3.png',0)
    ex, ey, et = classicalUtilitiesHS.derivative(img1, img2)
    print(ex)
    print(ey)
    print(et)
