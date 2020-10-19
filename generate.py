#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 11:33:08 2019

@author: Shuyi Li
"""
import tensorflow as tf
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import cv2
import tensorflow.contrib.layers as ly

def lrelu(x, leak=0.3, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)
        
def generator(z , reuse = True):
    train = ly.fully_connected(
        z, 4 * 4 * 512, activation_fn=lrelu, normalizer_fn=ly.batch_norm)
    train = tf.reshape(train, (-1, 4, 4, 512))
    #input ?*4*4*512 output 8*8*256(1)>23*23*128(2)>69*69*64(3)>69*69*1(4)
    train = ly.conv2d_transpose(train, 256, 3, stride=2,
                                activation_fn=tf.nn.relu, normalizer_fn=ly.batch_norm, padding='same', weights_initializer=tf.random_normal_initializer(0, 0.02))
    train = ly.conv2d_transpose(train, 128, 9, stride=2,
                                activation_fn=tf.nn.relu, normalizer_fn=ly.batch_norm, padding='valid', weights_initializer=tf.random_normal_initializer(0, 0.02))
    train = ly.conv2d_transpose(train, 64, 3, stride=3,
                                activation_fn=tf.nn.relu, normalizer_fn=ly.batch_norm, padding='SAME', weights_initializer=tf.random_normal_initializer(0, 0.02))
    train = ly.conv2d_transpose(train, channel, 3, stride=1,
                                activation_fn=tf.nn.tanh, padding='SAME', weights_initializer=tf.random_normal_initializer(0, 0.02))
    print(train.name)
    return train
    
def get_generator():
    z = tf.placeholder(dtype=tf.float32, shape=[batch_size, z_dim])
    with tf.variable_scope('generator'):
        train = generator(z)
    theta_g = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')

    return z, train
    
    
def generate_from_ckpt(a):
    with tf.device('/gpu:2'):
        z, train = get_generator()
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.6
    if ckpt_dir != None:
        with tf.Session(config=config) as sess:
            saver = tf.train.Saver()
            saver.restore(sess, './wgan/ckpt_wgan/model.ckpt-'+a)
            batch_z = np.random.normal(0, 1.0, [batch_size, z_dim]) \
                .astype(np.float32)
            rs = train.eval(feed_dict={z:batch_z})
    overall = []
    for i in range(8):
        temp = []
        for j in range(8):
            temp.append(rs[i * 8 + j])

        overall.append(np.concatenate(temp, axis=1))
    # change to the image shape
    res = np.concatenate(overall, axis=0)
    res = np.squeeze(res)
    res = (res+1)/2
    cv2.imwrite('sample-'+a, res)
    plt.figure(figsize=[8, 8])
    plt.imshow(res)
    plt.show()
    return res
