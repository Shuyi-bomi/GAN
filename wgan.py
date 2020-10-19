#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 16:48:00 2019
@author: Shuyi Li
"""

####### Code for WGAN & WGAN-GP
from __future__ import print_function
from preprocess import *
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"
from six.moves import xrange
import tensorflow.contrib.slim as slim
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import numpy as np
import tensorflow.contrib.layers as ly
from functools import partial
import scipy.io as scio
import math
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg # mpimg to read image
from PIL import Image

class WGAN:
    def __init__(self,X,max_iter_step = 10000,batch_size=64, is_mlp=False, is_adam = False, mode = 'gp'):
        '''
        X: data(#of observation* #of features), here is n*s*s*1 as image
        z_dim: dimension of random normal input
        s: img size(length of square image)
        Citers: update Citers times of critic in one iter(unless i < 25 or i % 500 == 0, i is iterstep)
        clamp_lower/clamp_upper: the upper bound and lower bound of parameters in critic
        is_mlp: whether to use mlp or dcgan stucture
        is_adam: whether to use adam for parameter update, if the flag is set False, use tf.train.RMSPropOptimizer as recommended in paper
        mode: 'gp' for gp WGAN and 'regular' for vanilla
        '''
        self.X = X
        self.max_iter_step = max_iter_step 
        self.batch_size = batch_size
        self.z_dim = 128
        self.learning_rate_ger = 5e-5
        self.learning_rate_dis = 5e-5
        self.device = '/gpu:0'
        self.s = X.shape[1]
        self.Citers = 5
        self.clamp_lower = -0.01
        self.clamp_upper = 0.01
        self.is_mlp = False
        self.is_adam = False
        # RGB/grey mode
        self.channel = 1
        self.mode = 'gp'
        # if 'gp' is chosen the corresponding lambda must be filled
        if self.mode=='gp':
            self.lam = 10.
        self.ngf = 64
        self.ndf = 64
        self.num_complete_minibatches = math.floor(self.X.shape[0]/self.batch_size)
        
    def random_mini_batches(self, mini_batch_size = 64, seed = 0):
        """
        Creates a list of random minibatches from (X, Y)

        Arguments:
        X -- input data, of shape (input size, number of examples)
        mini_batch_size -- size of the mini-batches, integer

        Returns:
        mini_batches -- list of synchronous (mini_batch_X)
        """

        np.random.seed()        # To make your "random" minibatches the same as ours, use np.random.seed(seed) 
        m = self.X.shape[0]                  # number of training examples
        mini_batches = []

        # Step 1: Shuffle (X, Y)
        permutation = list(np.random.permutation(m))
        shuffled_X = self.X[permutation,: ]

        # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
        num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
        for k in range(0, num_complete_minibatches):
            ### START CODE HERE ### (approx. 2 lines)
            mini_batch_X = shuffled_X[k * mini_batch_size : (k + 1) * mini_batch_size,:]
            ### END CODE HERE ###
            mini_batch = mini_batch_X
            mini_batches.append(mini_batch)

        return mini_batches

    def datarandom(self):
        #obtain random batch from shuffled dataset
        a= random.randint(0,self.num_complete_minibatches)    
        mini11=self.random_mini_batches()[a]
        return mini11
    def lrelu(x, leak=0.3, name="lrelu"):
        with tf.variable_scope(name):
            f1 = 0.5 * (1 + leak)
            f2 = 0.5 * (1 - leak)
            return f1 * x + f2 * abs(x)

    def generator_conv(self,z):
        train = ly.fully_connected(
            z, 4 * 4 * 512, activation_fn=self.lrelu, normalizer_fn=ly.batch_norm)
        train = tf.reshape(train, (-1, 4, 4, 512))
        #input n*4*4*512 output 8*8*256(1)>23*23*128(2)>69*69*64(3)>69*69*1(4)
        train = ly.conv2d_transpose(train, 256, 3, stride=2,
                                    activation_fn=tf.nn.relu, normalizer_fn=ly.batch_norm, padding='same', weights_initializer=tf.random_normal_initializer(0, 0.02))
        train = ly.conv2d_transpose(train, 128, 9, stride=2,
                                    activation_fn=tf.nn.relu, normalizer_fn=ly.batch_norm, padding='valid', weights_initializer=tf.random_normal_initializer(0, 0.02))
        train = ly.conv2d_transpose(train, 64, 3, stride=3,
                                    activation_fn=tf.nn.relu, normalizer_fn=ly.batch_norm, padding='SAME', weights_initializer=tf.random_normal_initializer(0, 0.02))
        train = ly.conv2d_transpose(train, channel, 3, stride=1,
                                    activation_fn=tf.nn.tanh, padding='SAME', weights_initializer=tf.random_normal_initializer(0, 0.02))
        #print(train.name)
        return train
        
    def generator_mlp(self, z):
    
        train = ly.fully_connected(
            z, 4 * 4 * 512, activation_fn=self.lrelulrelu, normalizer_fn=ly.batch_norm)
        train = ly.fully_connected(
            train, ngf, activation_fn=self.lrelulrelu, normalizer_fn=ly.batch_norm)
        train = ly.fully_connected(
            train, ngf, activation_fn=self.lrelulrelu, normalizer_fn=ly.batch_norm)
        train = ly.fully_connected(
            train, s*s*channel, activation_fn=tf.nn.tanh, normalizer_fn=ly.batch_norm)
        train = tf.reshape(train, tf.stack([batch_size, s, s, channel])) #batchsize*32*32*3
        return train
        
    def critic_conv(self, img, reuse=False):#69*69*1>23*23*64>12*12*128>4*4*256>2*2*512
        with tf.variable_scope('critic') as scope:
            if reuse:
                scope.reuse_variables()
            size = 64
            img = ly.conv2d(img, num_outputs=size, kernel_size=3,
                            stride=3, activation_fn=lrelu)
            img = ly.conv2d(img, num_outputs=size * 2, kernel_size=3,
                            stride=2,  padding='same', activation_fn=self.lrelu, normalizer_fn=ly.batch_norm)
            img = ly.conv2d(img, num_outputs=size * 4, kernel_size=3,
                            stride=3, activation_fn=self.lrelu, normalizer_fn=ly.batch_norm)
            img = ly.conv2d(img, num_outputs=size * 8, kernel_size=3,
                            stride=2, activation_fn=self.lrelu, normalizer_fn=ly.batch_norm)
            logit = ly.fully_connected(tf.reshape(
                img, [batch_size, -1]), 1, activation_fn=None)
        return logit
    
    def critic_mlp(self, img, reuse=False):
        with tf.variable_scope('critic') as scope:
            if reuse:
                scope.reuse_variables()
            size = 64
            img = ly.fully_connected(tf.reshape(
                img, [batch_size, -1]), ngf, activation_fn=tf.nn.relu)
            img = ly.fully_connected(img, ngf,
                activation_fn=tf.nn.relu)
            img = ly.fully_connected(img, ngf,
                activation_fn=tf.nn.relu)
            logit = ly.fully_connected(img, 1, activation_fn=None)
        return logit
        
    def build_graph(self):
        #build loss, optimizer for generator and critic
        noise_dist = tf.contrib.distributions.Normal(0., 1.)
        z = noise_dist.sample((self.batch_size, self.z_dim))
        if not self.is_mlp:
            generator =  self.generator_conv
            critic =  self.critic_conv
        else:
            generator =  self.generator_mlp
            critic =  self.critic_mlp
        with tf.variable_scope('generator'):
            train = generator(z)
        real_data = tf.placeholder(
            dtype=tf.float32, shape=(self.batch_size, self.s, self.s, self.channel))
        true_logit = critic(real_data)
        fake_logit = critic(train, reuse=True)
        c_loss = tf.reduce_mean(fake_logit - true_logit)
        if self.mode is 'gp':
            alpha_dist = tf.contrib.distributions.Uniform(low=0., high=1.)
            alpha = alpha_dist.sample((batch_size, 1, 1, 1))
            interpolated = real_data + alpha*(train-real_data)
            inte_logit = critic(interpolated, reuse=True)
            gradients = tf.gradients(inte_logit, [interpolated])[0]
           # gradients = tf.gradients(inte_logit, [interpolated,])[0]
            grad_l2 = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1,2,3]))
            gradient_penalty = tf.reduce_mean((grad_l2-1)**2)
            gp_loss_sum = tf.summary.scalar("gp_loss", gradient_penalty)
            grad = tf.summary.scalar("grad_norm", tf.nn.l2_loss(gradients))
            c_loss += lam*gradient_penalty
        g_loss = tf.reduce_mean(-fake_logit)
        g_loss_sum = tf.summary.scalar("g_loss", g_loss)
        c_loss_sum = tf.summary.scalar("c_loss", c_loss)
        img_sum = tf.summary.image("img", train, max_outputs=10)
        theta_g = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
        theta_c = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope='critic')
        counter_g = tf.Variable(trainable=False, initial_value=0, dtype=tf.int32)
        opt_g = ly.optimize_loss(loss=g_loss, learning_rate=self.learning_rate_ger,
                        optimizer=partial(tf.train.AdamOptimizer, beta1=0.5, beta2=0.9) if self.is_adam is True else tf.train.RMSPropOptimizer, 
                        variables=theta_g, global_step=counter_g,
                        summaries = ['gradient_norm'])
        counter_c = tf.Variable(trainable=False, initial_value=0, dtype=tf.int32)
        opt_c = ly.optimize_loss(loss=c_loss, learning_rate=self.learning_rate_dis,
                        optimizer=partial(tf.train.AdamOptimizer, beta1=0.5, beta2=0.9) if self.is_adam is True else tf.train.RMSPropOptimizer, 
                        variables=theta_c, global_step=counter_c,
                        summaries = ['gradient_norm'])
        if self.mode is 'regular':
            clipped_var_c = [tf.assign(var, tf.clip_by_value(var, clamp_lower, clamp_upper)) for var in theta_c]
            # merge the clip operations on critic variables
            with tf.control_dependencies([opt_c]):
                opt_c = tf.tuple(clipped_var_c)
        if not self.mode in ['gp', 'regular']:
            raise(NotImplementedError('Only two modes'))
        return opt_g, opt_c, real_data
        
    def main(self, log_dir = './log_wgan', ckpt_dir = './ckpt_wgan'):
    # run iteration
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)  
        with tf.device(device):
            opt_g, opt_c, real_data = self.build_graph()
        merged_all = tf.summary.merge_all()
        saver = tf.train.Saver()
        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.8
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
            for i in range(max_iter_step):
                dataset = self.datarandom()
                if i < 25 or i % 500 == 0:
                    citers = 100
                else:
                    citers = self.Citers
                for j in range(citers):
                    feed_dict = {real_data: dataset}
                    if i % 100 == 99 and j == 0:
                        run_options = tf.RunOptions(
                            trace_level=tf.RunOptions.FULL_TRACE)
                        run_metadata = tf.RunMetadata()
                        _, merged = sess.run([opt_c, merged_all], feed_dict=feed_dict,
                                             options=run_options, run_metadata=run_metadata)
                        summary_writer.add_summary(merged, i)
                        summary_writer.add_run_metadata(
                            run_metadata, 'critic_metadata {}'.format(i), i)
                    else:
                        sess.run(opt_c, feed_dict=feed_dict)                
                feed_dict = {real_data: dataset}
                if i % 100 == 99:
                    _, merged = sess.run([opt_g, merged_all], feed_dict=feed_dict,
                         options=run_options, run_metadata=run_metadata)
                    summary_writer.add_summary(merged, i)
                    summary_writer.add_run_metadata(
                        run_metadata, 'generator_metadata {}'.format(i), i)
                else:
                    sess.run(opt_g, feed_dict=feed_dict)                
                if i % 1000 == 999:
                    saver.save(sess, os.path.join(
                        ckpt_dir, "model.ckpt"), global_step=i)

                    
if __name__ == '__main__':
    data = getdata() #obtain data
    wgan = WGAN(data) #build class
    wgan.main() #run iterations 


