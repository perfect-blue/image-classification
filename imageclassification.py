# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 05:52:20 2020

@author: S430FN
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

#mask black and white
bw_img = cv2.imread('cat-example.jpg',1)
dimensions=bw_img.shape
height = bw_img.shape[0]
width=bw_img.shape[1]
channels=bw_img.shape[2]

print('Dimensions :',dimensions)
print('Image Height :',height)
print('Image Width :',width)
print('Number of Channels :',channels)

#window
win_row=4
win_col=4

for row in range(4):
    begin_row=int(row*height/4)
    end_row=int((row+1)*height/4)
    
    for col in range(4):
        begin_col=int(col*width/4)
        end_col=int((col+1)*width/4)
        
        mask = np.zeros(img.shape[:2],np.uint8)
        mask[begin_row:end_row,begin_col:end_col] = 255
        masked_img=cv2.bitwise_and(bw_img,bw_img,mask=mask)
        
#        plt.subplot(221), plt.imshow(bw_img, 'gray')
        plt.subplot(221), plt.imshow(masked_img, 'gray')        
  
        color=('b','g','r')
        for i, col in enumerate(color):
            histr=cv2.calcHist([img],[i],mask,[256],[0,256])
            plt.subplot(222), plt.plot(histr,color=col)
            plt.xlim([0,256])
    
        plt.show()








                
