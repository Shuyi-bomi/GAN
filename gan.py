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
    def __init__(self,mb_size = 128, Z_dim = 100, maxiter=5000):
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
    
    def lossopt(self, learning_rate=0.0001):
        #obtain loss for generator&discriminator as well as optimizer
        Z = tf.placeholder(tf.float32, shape=[None, 100], name='Z')
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
        
        return G_sample, D_loss, G_loss, D_optimizer, G_optimizer
    
    def random_mini_batches(self, X, mini_batch_size = self.mb_size, seed = 0):
        """
        Creates a list of random minibatches from (X)

        Arguments:
        X -- input data, of shape (input size, number of examples)
        mini_batch_size -- size of the mini-batches, integer

        Returns:
        mini_batches -- list of synchronous (mini_batch_X)
        """

        np.random.seed(seed)        # To make your "random" minibatches the same as ours, use np.random.seed(seed) 
        m = X.shape[0]                  # number of training examples
        mini_batches = []

        # Step 1: Shuffle (X)
        permutation = list(np.random.permutation(m))
        shuffled_X = X[permutation,: ]

        # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
        num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
        for k in range(0, num_complete_minibatches):
            ### START CODE HERE ### (approx. 2 lines)
            mini_batch_X = shuffled_X[k * mini_batch_size : (k + 1) * mini_batch_size,:]
            ### END CODE HERE ###
            mini_batch = mini_batch_X
            mini_batches.append(mini_batch)

        return mini_batches
    
    def sample_Z(self, m, n):
        '''Uniform prior for G(Z)'''
        return np.random.uniform(-1., 1., size=[m, n])
    
    def train(self, data_train):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            G_sample, D_loss, G_loss, D_optimizer, G_optimizer = self.lossopt()
            for it in range(self.maxiter):
                mini11=self.random_mini_batches(data_train)
                total_batch = math.ceil(data_train.shape[0]/self.mb_size)
            # iterate all batches
                for i in range(total_batch):
                    batch_x=mini11[i]                   
                    _, D_loss_curr = sess.run([D_optimizer, D_loss], feed_dict={
                                          X: batch_x, Z: sample_Z(self.mb_size, self.Z_dim)})
                    _, G_loss_curr = sess.run([G_optimizer, G_loss], feed_dict={
                                          Z: sample_Z(self.mb_size, self.Z_dim)})

                if it % 2000 == 0:
                    print('Iter: {}'.format(it))
                    print('D loss: {:.4}'.format(D_loss_curr))
                    print('G_loss: {:.4}'.format(G_loss_curr))
                    print()
                    samples = sess.run(G_sample, feed_dict={
                                       Z: sample_Z(16, self.Z_dim)})  # 16*4734
                    
if __name__ == '__main__':
    data = getdata(cnn=False) #obtain data
    gan = GAN() #build class
    gan.trian(data) #run iterations 
