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

    
#eksplorasi historgram
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
                histr=cv2.calcHist([image],[i],mask,[256],[0,256])
                plt.subplot(222), plt.plot(histr,color=col)
                plt.xlim([0,256])
    
            plt.show()    


#hitung dan normalisasi bins
def calculateBins(image,bins,mask):
    #menghitung histogram dan jumlah bins
    histB=cv2.calcHist([image],[0],mask,[bins],[0,256])
    histG=cv2.calcHist([image],[1],mask,[bins],[0,256])
    histR=cv2.calcHist([image],[2],mask,[bins],[0,256])
    
    #normalisasi
    normB=histB/len(histB)
    normG=histG/len(histG)
    normR=histR/len(histR)
    
    #concat jadi satu vektor
    bins_vector=[]
    for i in range(bins):
       temp=[float(normB[i]),float(normG[i]),float(normR[i])]
       bins_vector.append(temp)

        
    return bins_vector
        

from collections import Counter
def most_frequent(List):
    occurence_count = Counter(List)
    return occurence_count.most_common(1)[0][0]

kmeans = KMeans(n_clusters=16,random_state=0)

#menghitung bins untuk tiap windows
def calculateBinsByWindow(image):
    height = image.shape[0]
    width=image.shape[1]
    row_size=4
    col_size=4
    
    result=[]
    label=[]

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
            res=kmeans.labels_
            
            result.append(centroids)      
            label.append(res)
            
    return [result,label]
        

def feature_selection(centroid,labels):
    most_common_label=[]
    result=[]
    #cluster dengan anggota paling banyak
    for i in labels:
        most_common_label.append(most_frequent(i))
    for i in centroid:
        count=0
        result.append(i[most_common_label[count]])
        count+=1
    
    return result

from scipy.spatial import distance
#menghitung kesamaan fitur dengan euclidean distance
def feature_similarity(img1,img2):
    lab1=cv2.cvtColor(img1, cv2.COLOR_RGB2LAB)
    lab2=cv2.cvtColor(img1, cv2.COLOR_RGB2LAB)
    
    r1=calculateBinsByWindow(lab1)
    r2=calculateBinsByWindow(lab2)
    
    vector1=feature_selection(r1[0],r1[1])
    vector2=feature_selection(r2[0],r2[1])
    
    distances=[]
    for i in range(len(vector1)):
        d = distance.euclidean(vector1[i], vector2[i])
        distances.append(d)        
    
    print(distances)
    return [vector1,vector2]

kmeans.predict()
    
cat1 = cv2.imread('cat-example.jpg',3)
cat2 = cv2.imread('cat-example.jpg',3)
test=feature_similarity(cat1,cat2)
#countColor(cat1)
#countColor(cat2)
#img2=readImg('https://farm2.static.flickr.com/1347/930622888_4190f151bc.jpg')







    



                
