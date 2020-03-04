# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 05:52:20 2020

@author: S430FN
"""

import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from urllib.request import urlopen
import warnings
#warnings.filterwarnings("ignore")


    
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

cat1 = cv2.imread('cat-example.jpg',3)
conv1=cv2.cvtColor(cat1, cv2.COLOR_BGR2LAB) 
cv2.imshow('image',conv1)
cv2.waitKey(0)

#countColor(conv1)

#hitung dan normalisasi bins
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
        
bin_df=calculateBins(conv1,256,None)

from collections import Counter
def most_frequent(List):
    occurence_count = Counter(List)
    return occurence_count.most_common(1)[0][0]


kmeans = KMeans(n_clusters=20,random_state=20)

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
        
    count=0
    for i in centroid:
        result.append(i[most_common_label[count]])
        count+=1
    
#    print(most_common_label)
    return result


#menghitung kesamaan fitur dengan euclidean distance
def feature_similarity(img1):
    lab1=cv2.cvtColor(img1, cv2.COLOR_BGR2LAB)    
    r1=calculateBinsByWindow(lab1)    
    vector1=feature_selection(r1[0],r1[1])
    return np.array(vector1).reshape(-1)


fitur=[]
labels=[]


def insert_feature(path,file,label,jumlah):
    base_beach=path
    temp=file
    for i in range(jumlah):
        idx=i+1
        idx_conv=str(idx)
        name=temp+idx_conv+'.jpg'
        beach_img = cv2.imread(base_beach+name,3)
        vector = feature_similarity(beach_img)
        fitur.append(vector)
        labels.append(label)


            

cat2 = cv2.imread('piaget-2.jpg',3)

#masukan gambar-gambar
insert_feature("beach/","pantai","pantai",47)
print(np.array(fitur).shape)
insert_feature("hutan/","hutan","hutan",47)
insert_feature("snow/","snow","snow",44)
insert_feature("gedung/","gedung","gedung",44)

##classification
from sklearn import preprocessing

le=preprocessing.LabelEncoder()
img_en=le.fit_transform(labels)
arr_en=np.array(img_en).reshape(-1)

#KNN
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

y_real=['pantai','pantai','pantai','pantai','pantai',
        'gedung','gedung','gedung','gedung','gedung',
        'snow','snow','snow','snow','snow',
        'hutan','hutan','hutan','hutan','hutan']

data={'label_benar':y_real}    
dataframe=pd.DataFrame(data)

for i in range(1,17):
    model = KNeighborsClassifier(n_neighbors=i)
    model.fit(fitur,labels)
    
    y_pred=[]
    for j in range(1,21):
        index=j
        index_covt=str(index)
        name="test/test"+index_covt+".jpg"
        test_img=cv2.imread(name,3)
        test=[]
        pr=feature_similarity(test_img)
        test.append(np.array(pr).reshape(-1))
        
        #print(arr_pr.shape)
        ##arr_pr=np.array(pr).reshape(-1,1)
        predicted=model.predict(test)
        y_pred.append(predicted)
        
    #check akurasi
    y_result=np.array(y_pred).reshape(-1)
    print("akurasi untuk K=",i,": ",accuracy_score(y_result,y_real))
    dataframe['K='+str(i)]=y_result


print(dataframe)




    



                
