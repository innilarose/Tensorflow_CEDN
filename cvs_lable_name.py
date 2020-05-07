# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 13:50:36 2020

@author: Jino George
"""

import cv2
import numpy as np
import glob

label_lst=[]
file = open("Train_data.txt","w")
path = "O:\\innila\\Machine_Learning\\Edge_Detector_Vgg16\\tensorflow-object-contour-detection-master\\Dataset\\Portrait_Sample_Data\\*.*"
for img in glob.glob(path):
    img_path = img.split('\\')
    file_name = img_path[-1]
    img_name = file_name.split('.')
    label = img_name[0]
    print(label)
    file.write("%s \n" %label)
file.close()
    