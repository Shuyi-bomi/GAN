#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 11:25:04 2019

@author: apple
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import cv2 as cv
import math
import scipy.io as scio
from sklearn.utils import shuffle
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

class GAN:
    def __init__(self,mb_size = 128, Z_dim = 100, maxiter=500):
        '''
        mb_size: size of mini-batch
        Z_dim = size of random normal input
        maxiter: max number of iterations
        
        '''
        self.mb_size = mb_size
        self.Z_dim = Z_dim 
        self.maxiter = maxiter
        
    def weight_var(self, shape, name):
        return tf.get_variable(name=name, shape=shape, initializer=tf.contrib.layers.xavier_initializer())

    def bias_var(self, shape, name):
        return tf.get_variable(name=name, shape=shape, initializer=tf.constant_initializer(0))
        
    def generator(self, z):
        # generator net
        Z = tf.placeholder(tf.float32, shape=[None, 100], name='Z')
        G_W1 = self.weight_var([100, 128], 'G_W1')
        G_b1 = self.bias_var([128], 'G_B1')
        G_W2 = self.weight_var([128, 4734], 'G_W2')
        G_b2 = self.bias_var([4734], 'G_B2')
        theta_G = [G_W1, G_W2, G_b1, G_b2]
        
        G_h1 = tf.nn.relu(tf.matmul(z, G_W1) + G_b1)
        G_log_prob = tf.matmul(G_h1, G_W2) + G_b2
        G_prob = tf.nn.sigmoid(G_log_prob)
        
        return G_prob, theta_G
    
    def discriminator(self, x):
        # discriminater net
        X = tf.placeholder(tf.float32, shape=[None, 4734], name='X')
        D_W1 = self.weight_var([4734, 128], 'D_W1')
        D_b1 = self.bias_var([128], 'D_b1')
        D_W2 = self.weight_var([128, 1], 'D_W2')
        D_b2 = self.bias_var([1], 'D_b2')
        theta_D = [D_W1, D_W2, D_b1, D_b2]
        
        D_h1 = tf.nn.relu(tf.matmul(x, D_W1) + D_b1)
        D_logit = tf.matmul(D_h1, D_W2) + D_b2
        D_prob = tf.nn.sigmoid(D_logit)
        
        return D_prob, D_logit, theta_D
    
    def lossopt(self,learning_rate=0.0001):
        #obtain loss for generator&discriminator as well as optimizer
        G_sample,theta_G = self.generator(Z)
        D_real, D_logit_real, theta_D = self.discriminator(X)
        D_fake, D_logit_fake, _ = self.discriminator(G_sample)

        '''D_loss = -tf.reduce_mean(tf.log(D_real+ pow(10.0, -9)) + tf.log(1. - D_fake+ pow(10.0, -9)))
        G_loss = -tf.reduce_mean(tf.log(D_fake+ pow(10.0, -9)))'''

        D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_real, labels=tf.ones_like(D_logit_real)))
        D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.zeros_like(D_logit_fake)))
        D_loss = D_loss_real + D_loss_fake
        G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.ones_like(D_logit_fake)))

        D_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(D_loss, var_list=theta_D)
        G_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(G_loss, var_list=theta_G)

