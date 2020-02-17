# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 05:52:20 2020

@author: S430FN
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from urllib.request import urlopen

#membaca image dari url
def readImg(url):
    resp=urlopen(url)
    image = np.asarray(bytearray(resp.read()),dtype="uint8")
    image= cv2.imdecode(image,cv2.IMREAD_COLOR)
    
    return image

    
#fungsi untuk menghitung historgram
def countColor(image):
    dimensions=image.shape
    height = image.shape[0]
    width=image.shape[1]
    channels=image.shape[2]

    print('Dimensions :',dimensions)
    print('Image Height :',height)
    print('Image Width :',width)
    print('Number of Channels :',channels)

    for row in range(4):
        begin_row=int(row*height/4)
        end_row=int((row+1)*height/4)
    
        for col in range(4):
            begin_col=int(col*width/4)
            end_col=int((col+1)*width/4)
        
            mask = np.zeros(image.shape[:2],np.uint8)
            mask[begin_row:end_row,begin_col:end_col] = 255
            masked_img=cv2.bitwise_and(image,image,mask=mask)
        
            # plt.subplot(221), plt.imshow(bw_img, 'gray')
            plt.subplot(221), plt.imshow(masked_img, 'gray')        
  
            color=('b','g','r')
            for i, col in enumerate(color):
                histr=cv2.calcHist([image],[i],mask,[16],[0,256])
                plt.subplot(222), plt.plot(histr,color=col)
                plt.xlim([0,16])
    
            plt.show()    


#mengkategorikan pixel menjadi beberapa bins
def calculateBins(image,bins,mask):
    #menghitung histogram dan jumlah bins
    histB=cv2.calcHist([image],[0],mask,[bins],[0,256])
    histG=cv2.calcHist([image],[1],mask,[bins],[0,256])
    histR=cv2.calcHist([image],[2],mask,[bins],[0,256])
    
    #normalisasi
    normB=histB/max(histB)
    normG=histG/max(histG)
    normR=histR/max(histR)
    
    #concat jadi satu vektor
    bins_vector=[]
    for i in range(bins):
       temp=[float(normB[i]),float(normG[i]),float(normR[i])]
       bins_vector.append(temp)

        
    return bins_vector
        


def calculateBinsByWindow(image):
    height = image.shape[0]
    width=image.shape[1]
    row_size=4
    col_size=4
    
    result=[]
    kmeans = KMeans(n_clusters=20,random_state=None)
    for row in range(row_size):
        begin_row=int(row*height/row_size)
        end_row=int((row+1)*height/row_size)
    
        for col in range(col_size):
            begin_col=int(col*width/4)
            end_col=int((col+1)*width/4)
        
            mask = np.zeros(image.shape[:2],np.uint8)
            mask[begin_row:end_row,begin_col:end_col] = 255
            data =calculateBins(image,256,mask)
            kmeans.fit(data)
            
            centroids=kmeans.cluster_centers_
            result.append(centroids)      
    
    return result
        


    
img = cv2.imread('cat-example.jpg',3)
img2=readImg('https://farm2.static.flickr.com/1347/930622888_4190f151bc.jpg')

conv= cv2.cvtColor(img2, cv2.COLOR_RGB2LAB)
#countColor(conv)
#calculateBins(conv,20,None)
cluster_center=calculateBinsByWindow(conv)




                
