#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 16:48:00 2019

@author: Shuyi Li
"""


#%%
from __future__ import print_function
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import scipy.io as scio
 
#%%
def getdata():
    dataFile = './data_planet.mat'
    data = scio.loadmat(dataFile)
    data_train = np.transpose(data['data_single'])
    mm = MinMaxScaler()
    data_train=mm.fit_transform(data_train)
    #4734>69*69=4761, pad to desired shape
    data_train1 = np.pad(data_train,((0,0),(14,13)),'constant',constant_values=(0,0))
    data_train1 = np.reshape(data_train1, (-1, 69, 69)) #none*69*69
    data_train1 = np.expand_dims(data_train1, -1) #generate dim=4, none*69*69*1
    data_train1 = data_train1.astype(np.float32)
    return data_train1
