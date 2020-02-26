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
            data =calculateBins(image,255,mask)
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
#    lab2=cv2.cvtColor(img2, cv2.COLOR_BGR2LAB)
    
    r1=calculateBinsByWindow(lab1)
#    r2=calculateBinsByWindow(lab2)
    
    vector1=feature_selection(r1[0],r1[1])
#    vector2=feature_selection(r2[0],r2[1])
    
    
#    distances=[]
#    res=0
#    for i in range(len(vector1)):
#        d = distance.euclidean(vector1[i], vector2[i])
#        res+=d
#        distances.append(d)        
#    
#    
#    print(vector1)
#    threshold.append(res)
    return np.array(vector1).reshape(-1)
#    return vector1

fitur=[]
labels=[]
debug=[]


def insert_feature(path,file,label):
    base_beach=path
    temp=file
    for i in range(22):
        idx=i+1
        idx_conv=str(idx)
        name=temp+idx_conv+'.jpg'
        beach_img = cv2.imread(base_beach+name,3)
        vector = feature_similarity(beach_img)
        fitur.append(vector)
        labels.append(label)


    
cat1 = cv2.imread('pantai-test.jpg',3)
cat2 = cv2.imread('piaget-2.jpg',3)
#test=feature_similarity(cat1)
insert_feature("beach/","pantai","pantai")
print(np.array(fitur).shape)

#debug=[]
#debug.append([1,2,3])
#debug.append([1,2,3])
#print(np.array(debug).shape)
insert_feature("hutan/","hutan","hutan")

##classification
from sklearn import preprocessing
#
le=preprocessing.LabelEncoder()
img_en=le.fit_transform(labels)
arr_en=np.array(img_en).reshape(-1)

#KNN
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

model = KNeighborsClassifier(n_neighbors=6)
X_train, X_test, y_train, y_test = train_test_split(fitur, labels, test_size=0.3)
model.fit(fitur,labels)

#print(model.predict(X_test))
#
test=[]
pr=feature_similarity(cat1)
test.append(np.array(pr).reshape(-1))

#print(arr_pr.shape)
##arr_pr=np.array(pr).reshape(-1,1)
predicted=model.predict(test)
print(predicted)







    



                
