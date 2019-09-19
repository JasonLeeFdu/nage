import  tensorflow as tf
import  tensorboard

# Traditional
import os
import numpy as np
import time
import cv2 as cv
from datetime import datetime
import math
from tqdm import tqdm

# Program
import Config as conf
import NetBricks.ljchopt1 as bricks
import Utils.common as tools

# network

from network import  network
from tensorflow.contrib.slim import nets






with tf.Session() as sess:
    li = list()
    logits_28 = tf.random_normal([5,28,28,5])
    logits_56 = tf.random_normal([5,56,56,5])
    logits_112 = tf.random_normal([5,112,112,5])
    logits_224 = tf.random_normal([5,224,224,5])
    logitsClass = tf.random_normal([5,1,1,5])
    gtImg = tf.random_normal([5,224,224,5])
    gtLb = tf.random_normal([5,5])
    '''
    z = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=gtImg,logits=logits_224,dim=3))
    gtImgg = tf.image.resize_bicubic(gtImg,[112,112])
    zz =  tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=gtImgg,logits=logits_112,dim=3))
    '''

    gtLb = tf.expand_dims(gtLb,axis=1);gtLb = tf.expand_dims(gtLb,axis=1)

    zz_ = sess.run(gtLb)


    p = 43




























'''
 li.append(logits_28)
    li.append(logits_56)
    li.append(logits_112)
    li.append(logits_224)
    li.append(logitsClass)
    li.append(gtImg)
    li.append(gtLb)

'''