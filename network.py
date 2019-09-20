## image IO
import imageio
import skimage

# tf
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
import Utils.common as tools

# network

from tensorflow.contrib.slim import nets
from resTool.ResNetGen import *

slim = tf.contrib.slim
dbgList = list()

def ConvLayer(inp,h,w,inc,outc,training,padding='SAME',strides=[1,1,1,1],name='Conv2d'):
    with tf.name_scope(name):
        weight = tf.Variable(tf.truncated_normal([h,w,inc,outc],mean=0,stddev=1e-3),name='weight')
        bias   = tf.Variable(tf.truncated_normal([outc],mean=0,stddev=1e-8),name='bias')
        out    = tf.nn.conv2d(inp,weight,padding=padding,name='conv',strides=strides) + bias
        out    = tf.layers.batch_normalization(out,training=training)
        out    = tf.nn.relu(out)
    return out

def ConvLayerD(inp,h,w,inc,outc,training,dilated = [1,1,1,1],padding='SAME',strides=[1,1,1,1],name='Conv2d'):
    with tf.name_scope(name):
        weight = tf.Variable(tf.truncated_normal([h, w, inc, outc], mean=0, stddev=1e-3), name='weight')
        bias   = tf.Variable(tf.truncated_normal([outc],mean=0,stddev=1e-8),name='bias')
        out    = tf.nn.atrous_conv2d(inp,weight,2,padding='SAME') + bias
        out    = tf.layers.batch_normalization(out,training=training)
        out    = tf.nn.relu(out)
    return out

def ConvLayerNoRELU(inp,h,w,inc,outc,training,padding='SAME',strides=[1,1,1,1],name='Conv2d'):
    with tf.name_scope(name):
        weight = tf.Variable(tf.truncated_normal([h,w,inc,outc],mean=0,stddev=1e-3),name='weight')
        bias   = tf.Variable(tf.truncated_normal([outc],mean=0,stddev=1e-8),name='bias')
        out    = tf.nn.conv2d(inp,weight,padding=padding,name='conv',strides=strides) + bias
        out = tf.layers.batch_normalization(out, training=training)
    return out

def bridgeConv1(inp,training,name='BridgeConv1'):
    with tf.name_scope(name):
        l2 = ConvLayer(inp, 3, 3, 64, 32, training, name='Conv2')
    return l2


'''
def bridgeOri(inp,training,name='BridgeOri'):
    with tf.name_scope(name):
        l1 = ConvLayer(inp,3,3,3,32,training,name='Conv1')
        l2 = ConvLayer(l1,3,3,32,32,training,name='Conv2')
        l3 = ConvLayer(l2,3,3,32,32,training,name='Conv3')
    return l3 + l1
'''

def bridgeOri(inp,training,name='BridgeOri'):
    with tf.name_scope(name):
        l1 = ConvLayer(inp,3,3,64,32,training,name='Conv1')
        l2 = ConvLayerD(l1,3,3,32,32,training,dilated=[1,2,2,1],name='Conv2')
        l3 = ConvLayer(l2,3,3,32,16,training,name='Conv3')
    return l3

def bridgeOriRes18(inp,training,name='BridgeOri'):
    with tf.name_scope(name):
        l1 = ConvLayer(inp,3,3,32,32,training,name='Conv1')
        l2 = ConvLayerD(l1,3,3,32,32,training,dilated=[1,2,2,1],name='Conv2')
        l3 = ConvLayer(l2,3,3,32,16,training,name='Conv3')
    return l3


def horzBlock(inp,inc,outc,trainingFlag,name='horzBlock'):
    with tf.name_scope(name):
        l1 = ConvLayer(inp,3,3,inc,outc,trainingFlag)
        l2 = ConvLayer(l1,3,3,outc,outc,trainingFlag)
    return inp + l2

def vertBlock(inp,inc,trainingFlag,name='vertBlock'):
    with tf.name_scope(name):
        outc = inc // 2
        l1 = ConvLayer(inp,1,1,inc,inc,trainingFlag)
        l2 = tf.layers.conv2d_transpose(l1,outc,3,(2,2),activation=tf.nn.relu,padding='SAME')
        l3 = ConvLayer(l2,3,3,outc,outc,trainingFlag)
    return l3

###
def ResNet50Eye(inp,trainingFlag):
    with slim.arg_scope(nets.resnet_v2.resnet_arg_scope()):
        resnet50,endPoints = nets.resnet_v2.resnet_v2_50(inp,num_classes=5,is_training=trainingFlag)
        conv1 = endPoints['resnet_v2_50/conv1']
        block1Unit2 = endPoints['resnet_v2_50/block1/unit_2/bottleneck_v2'] # [56,56,256]##
        block1Out = endPoints['resnet_v2_50/block1'] #[28,28,256]##
        block2Out = endPoints['resnet_v2_50/block2'] #[14,14,512]
        block3Out = endPoints['resnet_v2_50/block3'] #[7,7,1024]
        block4Out = endPoints['resnet_v2_50/block4'] #[7,7,2048]
        with tf.name_scope('FPN_Pyramid'):
            # 特征融合
            bridgeL1 = bridgeOri(inp, trainingFlag)  # [224,224,32]##
            bridgeL2 = bridgeConv1(conv1, trainingFlag)  # [224,224,32]
            block4_ = ConvLayer(block4Out,1,1,2048,1024,training=trainingFlag)
            block3_ = ConvLayer(block3Out,1,1,1024,1024,training=trainingFlag)
            block2_ = ConvLayer(block2Out,1,1,512,512,training=trainingFlag)
            block1_ = ConvLayer(block1Out,1,1,256,256,training=trainingFlag)
            blockU2 = ConvLayer(block1Unit2,1,1,256,128,training=trainingFlag)
            rightL4 = block3_ + block4_
            rightL2 = tf.layers.conv2d_transpose(rightL4,512,3,(2,2),activation=tf.nn.relu,padding='SAME')
            rightL2 = block2_ + rightL2
            rightL1 = tf.layers.conv2d_transpose(rightL2,256,3,(2,2),activation=tf.nn.relu,padding='SAME')
            rightL1_28 = tf.add(block1_ , rightL1,name='right_28')         # =>  #  FPN_Pyramid/right_28:0
            rightL1_28_ = tf.layers.conv2d_transpose(rightL1_28, 128, 3, (2, 2),activation=tf.nn.relu, padding='SAME')
            rightL1_56 = tf.add(rightL1_28_ , blockU2,name='right_56')         # =>  # FPN_Pyramid/right_56:0
            rightL1_56_ = ConvLayer(rightL1_56,3,3,128,64,training=trainingFlag,name='rightL1_56_')
            rightL0 = tf.layers.conv2d_transpose(rightL1_56_, 32, 3, (2, 2),activation=tf.nn.relu, padding='SAME')
            rightL0_112 = tf.add(rightL0, bridgeL2, name='right_112')           # =>  # FPN_Pyramid/right_112:0
            rightL = tf.layers.conv2d_transpose(rightL0_112, 32, 3, (2, 2),activation=tf.nn.relu, padding='SAME')
            rightL_224 = tf.add(rightL,bridgeL1,name='right_224')               # =>  # FPN_Pyramid/right_224:0
            # outward opts
            logits_28 = ConvLayer(rightL1_28,3,3,256,conf.FIANL_CLASSES_NUM,training=trainingFlag,name='logits_28')
            logits_56 = ConvLayer(rightL1_56,3,3,128,conf.FIANL_CLASSES_NUM,training=trainingFlag,name='logits_56')
            logits_112= ConvLayer(rightL0_112,3,3,32,conf.FIANL_CLASSES_NUM,training=trainingFlag,name='logits_112')
            logits_224= ConvLayer(rightL_224,3,3,32,conf.FIANL_CLASSES_NUM,training=trainingFlag,name='logits_224')
            logitsClass = endPoints['resnet_v2_50/logits'] #(5, 1, 1, 5)
            predFlat = tf.argmax(logits_224, axis=3)
            predVis =  tf.nn.softmax(logits_224,axis=3)
            predVis = predVis[:,:,:,1]
            predCls = tf.squeeze(tf.argmax(logitsClass,axis=3))

            ## 观察网路的参数变化情况 -- 仅仅在网络内部加入histogram 观察参数分布的变化
            dg = tf.get_default_graph()
            tf.summary.histogram('logits_224/weights',dg.get_tensor_by_name('FPN_Pyramid/logits_224/weight:0'))
            tf.summary.histogram('logits_224/bias',dg.get_tensor_by_name('FPN_Pyramid/logits_224/bias:0'))
            tf.summary.histogram('logits_28/weights',dg.get_tensor_by_name('FPN_Pyramid/logits_28/weight:0'))
            tf.summary.histogram('logits_28/bias',dg.get_tensor_by_name('FPN_Pyramid/logits_28/bias:0'))
            tf.summary.histogram('resnet_v2_50/logits/weights',dg.get_tensor_by_name('resnet_v2_50/logits/weights:0'))
            tf.summary.histogram('resnet_v2_50/logits/bias',dg.get_tensor_by_name('resnet_v2_50/logits/biases:0'))
        return logits_28,logits_56,logits_112,logits_224,logitsClass,predFlat,predCls,predVis

### 实验版
def vgg19Eye(inp, trainingFlag):
    with slim.arg_scope(nets.vgg.vgg_arg_scope()):
        resnet50, endPoints = nets.vgg.vgg_19(inp,num_classes=conf.FIANL_CLASSES_NUM,is_training=trainingFlag)
        vgg224 = endPoints['vgg_19/conv1/conv1_2']
        vgg112 = endPoints['vgg_19/pool1']
        vgg56 = endPoints['vgg_19/pool2']
        vgg28 = endPoints['vgg_19/pool3']
        logitMid1 = endPoints['vgg_19/pool5']
        with tf.name_scope('FPN_Pyramid'):
            logitMid2 = ConvLayer(logitMid1, 7, 7, 512, 256, trainingFlag, 'VALID')
            logitMid2 = dropout(logitMid2, trainingFlag)
            logitFinal = fully_conneted(logitMid2, conf.FIANL_CLASSES_NUM, scope='logitFinal')


            pipe224 = bridgeOri(vgg224,trainingFlag)
            pipe112 = ConvLayer(vgg112,3,3,64,32,trainingFlag)
            pipe56 = ConvLayer(vgg56,3,3,128,64,trainingFlag)
            pipe28 = ConvLayer(vgg28,3,3,256,128,trainingFlag)              #=>[None,28,28,128]

            fm56New = tf.image.resize_images(pipe28,[56,56],align_corners=True)
            fm56New = ConvLayer(fm56New,3,3,128,64,trainingFlag)
            out56 = fm56New + pipe56                                        #=>[None,56,56,64]
            fm112New = tf.layers.conv2d_transpose(out56,32,3,(2,2),activation=tf.nn.relu,padding='SAME')
            out112 = fm112New + pipe112                                     #=>[None,112,112,32]
            fm224New = tf.layers.conv2d_transpose(out112,16,3,(2,2),activation=tf.nn.relu,padding='SAME')
            out224 = fm224New + pipe224                                     #=>[None,224,224,16]

            # outward opts
            # outward opts
            logits_28 = ConvLayerNoRELU(pipe28, 1, 1, 128, conf.FIANL_CLASSES_NUM, training=trainingFlag,
                                        name='logits_28')
            logits_56 = ConvLayerNoRELU(out56, 1, 1, 64, conf.FIANL_CLASSES_NUM, training=trainingFlag,
                                        name='logits_56')
            logits_112 = ConvLayerNoRELU(out112, 1, 1, 32, conf.FIANL_CLASSES_NUM, training=trainingFlag,
                                         name='logits_112')
            logits_224 = ConvLayerNoRELU(out224, 1, 1, 16, conf.FIANL_CLASSES_NUM, training=trainingFlag,
                                         name='logits_224')
            logits_28_224 = tf.image.resize_images(logits_28, [224, 224], align_corners=True)
            logits_56_224 = tf.image.resize_images(logits_56, [224, 224], align_corners=True)
            logits_112_224 = tf.image.resize_images(logits_112, [224, 224], align_corners=True)
            concatedLogits = tf.concat([logits_28_224, logits_56_224, logits_112_224, logits_224], axis=-1)
            logitMask = ConvLayerNoRELU(concatedLogits, 1, 1, 4 * conf.FIANL_CLASSES_NUM, conf.FIANL_CLASSES_NUM,
                                        trainingFlag)

            logitsClass =logitFinal
            predFlat = tf.argmax(logitMask, axis=3)
            predVis = tf.nn.softmax(logitMask, axis=3)
            predVis = predVis[:, :, :, 1]
            predCls = tf.argmax(logitsClass, axis=1)
            logitsClass = tf.expand_dims(logitsClass,axis=1)
            logitsClass = tf.expand_dims(logitsClass,axis=1)
            ## 观察网路的参数变化情况 -- 仅仅在网络内部加入histogram 观察参数分布的变化
            dg = tf.get_default_graph()
            tf.summary.histogram('logits_224/weights', dg.get_tensor_by_name('FPN_Pyramid/logits_224/weight:0'))
            tf.summary.histogram('logits_224/bias', dg.get_tensor_by_name('FPN_Pyramid/logits_224/bias:0'))
        tf.summary.image('predVisBin', tf.expand_dims(tf.cast(predVis > 0.5, tf.float32), axis=-1))

        return logits_28, logits_56, logits_112, logits_224, logitMask, logitsClass, predFlat, predCls, predVis


### 测试版v1 2019年9月20号周五
def ResNet18Eyev1(inp, trainingFlag):
    endPoints = getRes18(inp,conf.FIANL_CLASSES_NUM,trainingFlag)
    L224 = endPoints['L224']
    L112 = endPoints['L112']
    L56 = endPoints['L56']
    L28 = endPoints['L28']
    logitFinal = endPoints['logitFinal']

    with tf.name_scope('FPN_Pyramid'):
        pipe224 = bridgeOriRes18(L224, trainingFlag)
        pipe112 = ConvLayer(L112, 3, 3, 64, 32, trainingFlag)
        pipe56 = ConvLayer(L56, 3, 3, 128, 64, trainingFlag)
        pipe28 = ConvLayer(L28, 3, 3, 256, 128, trainingFlag)  # =>[None,28,28,128]

        fm56New = tf.image.resize_images(pipe28, [56, 56], align_corners=True)
        fm56New = ConvLayer(fm56New, 3, 3, 128, 64, trainingFlag)
        out56 = fm56New + pipe56  # =>[None,56,56,64]
        fm112New = tf.layers.conv2d_transpose(out56, 32, 3, (2, 2), activation=tf.nn.relu, padding='SAME')
        out112 = fm112New + pipe112  # =>[None,112,112,32]
        fm224New = tf.layers.conv2d_transpose(out112, 16, 3, (2, 2), activation=tf.nn.relu, padding='SAME')
        out224 = fm224New + pipe224  # =>[None,224,224,16]

        # outward opts
        logits_28 = ConvLayerNoRELU(pipe28, 1, 1, 128, conf.FIANL_CLASSES_NUM, training=trainingFlag,
                              name='logits_28')
        logits_56 = ConvLayerNoRELU(out56, 1, 1, 64, conf.FIANL_CLASSES_NUM, training=trainingFlag,
                              name='logits_56')
        logits_112 = ConvLayerNoRELU(out112, 1, 1, 32, conf.FIANL_CLASSES_NUM, training=trainingFlag,
                               name='logits_112')
        logits_224 = ConvLayerNoRELU(out224, 1, 1, 16, conf.FIANL_CLASSES_NUM, training=trainingFlag,
                               name='logits_224')
        logits_28_224 = tf.image.resize_images(logits_28,[224,224],align_corners=True)
        logits_56_224 = tf.image.resize_images(logits_56,[224,224],align_corners=True)
        logits_112_224 = tf.image.resize_images(logits_112,[224,224],align_corners=True)
        concatedLogits = tf.concat([logits_28_224,logits_56_224,logits_112_224,logits_224],axis=-1)
        logitMask = ConvLayerNoRELU(concatedLogits,1,1,4*conf.FIANL_CLASSES_NUM,conf.FIANL_CLASSES_NUM,trainingFlag)

        logitsClass = tf.expand_dims(logitFinal, axis=1)
        logitsClass = tf.expand_dims(logitsClass, axis=1)
        predFlat = tf.argmax(logitMask, axis=3)
        predVis = tf.nn.softmax(logitMask, axis=3)
        predVis = predVis[:, :, :, 1]
        predCls = tf.squeeze(tf.argmax(logitsClass, axis=3))

        ## 观察网路的参数变化情况 -- 仅仅在网络内部加入histogram 观察参数分布的变化
        dg = tf.get_default_graph()
        tf.summary.histogram('logits_224/weights', dg.get_tensor_by_name('FPN_Pyramid/logits_224/weight:0'))
        tf.summary.histogram('logits_224/bias', dg.get_tensor_by_name('FPN_Pyramid/logits_224/bias:0'))
        tf.summary.histogram('Resnet/logitFinal/dense/kernel:0', dg.get_tensor_by_name('Resnet/logitFinal/dense/kernel:0'))
        tf.summary.histogram('Resnet/logitFinal/dense/bias:0', dg.get_tensor_by_name('Resnet/logitFinal/dense/bias:0'))
    tf.summary.image('predVisBin', tf.expand_dims(tf.cast(predVis > 0.5,tf.float32),axis=-1))


    return logits_28, logits_56, logits_112, logits_224, logitMask, logitsClass, predFlat, predCls, predVis


### 实验版
def vgg16Eye(inp, trainingFlag):
    with slim.arg_scope(nets.vgg.vgg_arg_scope()):
        resnet50, endPoints = nets.vgg.vgg_16(inp,num_classes=conf.FIANL_CLASSES_NUM,is_training=trainingFlag)
        vgg224 = endPoints['vgg_16/conv1/conv1_2']
        vgg112 = endPoints['vgg_16/pool1']
        vgg56 = endPoints['vgg_16/pool2']
        vgg28 = endPoints['vgg_16/pool3']
        with tf.name_scope('FPN_Pyramid'):
            pipe224 = bridgeOri(vgg224,trainingFlag)
            pipe112 = ConvLayer(vgg112,3,3,64,32,trainingFlag)
            pipe56 = ConvLayer(vgg56,3,3,128,64,trainingFlag)
            pipe28 = ConvLayer(vgg28,3,3,256,128,trainingFlag)              #=>[None,28,28,128]

            fm56New = tf.image.resize_images(pipe28,[56,56],align_corners=True)
            fm56New = ConvLayer(fm56New,3,3,128,64,trainingFlag)
            out56 = fm56New + pipe56                                        #=>[None,56,56,64]
            fm112New = tf.layers.conv2d_transpose(out56,32,3,(2,2),activation=tf.nn.relu,padding='SAME')
            out112 = fm112New + pipe112                                     #=>[None,112,112,32]
            fm224New = tf.layers.conv2d_transpose(out112,16,3,(2,2),activation=tf.nn.relu,padding='SAME')
            out224 = fm224New + pipe224                                     #=>[None,224,224,16]

            # outward opts
            logits_28 = ConvLayer(pipe28, 1, 1, 128, conf.FIANL_CLASSES_NUM, training=trainingFlag,
                                  name='logits_28')
            logits_56 = ConvLayer(out56, 1, 1, 64, conf.FIANL_CLASSES_NUM, training=trainingFlag,
                                  name='logits_56')
            logits_112 = ConvLayer(out112, 1, 1, 32, conf.FIANL_CLASSES_NUM, training=trainingFlag,
                                   name='logits_112')
            logits_224 = ConvLayer(out224, 1, 1, 16, conf.FIANL_CLASSES_NUM, training=trainingFlag,
                                   name='logits_224')

            logits_28_224 = tf.image.resize_images(logits_28, [224, 224], align_corners=True)
            logits_56_224 = tf.image.resize_images(logits_56, [224, 224], align_corners=True)
            logits_112_224 = tf.image.resize_images(logits_112, [224, 224], align_corners=True)
            concatedLogits = tf.concat([logits_28_224, logits_56_224, logits_112_224, logits_224])
            logitMask = ConvLayer(concatedLogits, 1, 1, 4 * conf.FIANL_CLASSES_NUM, conf.FIANL_CLASSES_NUM,
                                  trainingFlag)


            logitsClass = endPoints['vgg_16/fc8']  # (5, 1, 1, 5)





            logitsClass = tf.expand_dims(logitsClass,axis=1)
            logitsClass = tf.expand_dims(logitsClass,axis=1)
            predFlat = tf.argmax(logitMask, axis=3)
            predVis = tf.nn.softmax(logitMask, axis=3)
            predVis = predVis[:, :, :, 1]
            predCls = tf.squeeze(tf.argmax(logitsClass, axis=3))

            ## 观察网路的参数变化情况 -- 仅仅在网络内部加入histogram 观察参数分布的变化
            dg = tf.get_default_graph()
            tf.summary.histogram('logits_224/weights', dg.get_tensor_by_name('FPN_Pyramid/logits_224/weight:0'))
            tf.summary.histogram('logits_224/bias', dg.get_tensor_by_name('FPN_Pyramid/logits_224/bias:0'))
            tf.summary.histogram('logits_28/weights', dg.get_tensor_by_name('FPN_Pyramid/logits_28/weight:0'))
            tf.summary.histogram('logits_28/bias', dg.get_tensor_by_name('FPN_Pyramid/logits_28/bias:0'))
            tf.summary.histogram('vgg_16/fc8/weights', dg.get_tensor_by_name('vgg_16/fc8/weights:0'))
            tf.summary.histogram('vgg_16/fc7/weights', dg.get_tensor_by_name('vgg_16/fc7/weights:0'))

            return logits_28, logits_56, logits_112, logits_224, logitMask, logitsClass, predFlat, predCls, predVis





'''
以下是损失函数类


'''
def lossFunc(logits_28,logits_56,logits_112,logits_224, logitMask, logitsClass,gtImg,gtLb):
    # gtImg  [None,224,224,numClass];   gtLb [None,Numclass]
    ######################logits_28, logits_56, logits_112, logits_224, logitsClass, label, clsLabel)
    # logits [None,x,x,numClass]    logitsClass  [None,1,1,Numclass]
    '''
    logits_mask_safe = tf.clip_by_value(logitMask,1e-10,10)
    logits_224_safe = tf.clip_by_value(logits_224,1e-10,10)
    logits_112_safe = tf.clip_by_value(logits_112,1e-10,10)
    logits_56_safe = tf.clip_by_value(logits_56,1e-10,10)
    logits_28_safe = tf.clip_by_value(logits_28,1e-10,10)
    logitsClass_safe = tf.clip_by_value(logitsClass,1e-10,10)
    '''

    logits_mask_safe = clipSmall(logitMask)
    logits_224_safe = clipSmall(logits_224)
    logits_112_safe = clipSmall(logits_112)
    logits_56_safe = clipSmall(logits_56)
    logits_28_safe = clipSmall(logits_28)
    logitsClass_safe = clipSmall(logitsClass)




    Loss_Mask = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=gtImg, logits=logits_mask_safe, dim=3))
    Loss224 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=gtImg, logits=logits_224_safe, dim=3))
    gtImg_112 = tf.image.resize_bicubic(gtImg, [112, 112])
    Loss_112 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=gtImg_112, logits=logits_112_safe, dim=3))
    gtImg_56 = tf.image.resize_bicubic(gtImg_112, [56, 56])
    Loss_56  = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=gtImg_56, logits=logits_56_safe, dim=3))
    gtImg_28 = tf.image.resize_bicubic(gtImg_56, [28, 28])
    Loss_28  = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=gtImg_28, logits=logits_28_safe, dim=3))


    Loss_Cls = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=gtLb, logits=logitsClass_safe, dim=3))
    lossMask = 1.3*Loss_Mask + Loss224 + Loss_112 + Loss_56 + Loss_28

    loss = lossMask + Loss_Cls* 1.3
    #
    predCls = tf.squeeze(tf.argmax(logitsClass_safe, axis=3))
    gtCls = tf.squeeze(tf.argmax(gtLb, axis=3))
    N = tf.reduce_sum(tf.cast(tf.equal(gtCls, gtCls), tf.float32))
    a = tf.reduce_sum(tf.cast(tf.equal(predCls, gtCls), tf.float32))
    precision = a / N


    tf.summary.scalar('Loss224',Loss224)
    tf.summary.scalar('LossMask224',Loss_Mask)
    tf.summary.scalar('Loss_112',Loss_112)
    tf.summary.scalar('Loss_56',Loss_56)
    tf.summary.scalar('Loss_28',Loss_28)
    tf.summary.scalar('Loss_Cls',Loss_Cls)
    tf.summary.scalar('Precision',precision)


    loss = tf.clip_by_value(loss,0,100.0)


    return loss

def loadPretrainedResnet50(sess):
    vgg_var_list = tf.global_variables('resnet_v2_50')
    vgg_var_list = vgg_var_list[:-2]
    saver = tf.train.Saver(vgg_var_list)
    saver.restore(sess, 'model/pretrained/resnet_v2_50.ckpt')
    print('Pretrained Loaded')

def loadPretrainedResnetVGG19(sess):
    vgg_var_list = tf.global_variables('vgg_19')
    vgg_var_list = vgg_var_list[:-2]
    saver = tf.train.Saver(vgg_var_list)
    saver.restore(sess, 'model/pretrained/vgg_19.ckpt')
    print('Pretrained Loaded')

def clipSmall(logits):
    posLo = tf.cast(logits >= 0,tf.float32)
    negLo = tf.cast(logits < 0,tf.float32)
    posPart = posLo * logits
    negPart = negLo * logits
    posGood = tf.clip_by_value(posPart,1e-10,10)
    negGood = tf.clip_by_value(negPart,-10,-(1e-10))

    return posGood + negGood

def focalLossMap(output, label):
    '''
    :param output:[None,H,W,conf.TOTAL_CLS]
    :param label:[[None,H,W,conf.TOTAL_CLS]]
    :return:[]
    '''


    return tf.reduce_sum(output)

def focalLossCls(output, label):
    label = tf.cast(tf.greater(label, 0.5), tf.float32)

    num_labels_pos = tf.reduce_sum(label)
    num_labels_neg = tf.reduce_sum(1.0 - label)
    num_total = num_labels_pos + num_labels_neg

    p = tf.sigmoid(output)
    # p = output
    pos = tf.multiply(p, label)
    neg = tf.multiply((1.0 - p), (1.0 - label))
    final_p = pos + neg
    pos_p = tf.clip_by_value(pos, 1e-12, (1.0 - 1e-12))
    neg_p = tf.clip_by_value(neg, 1e-12, (1.0 - 1e-12))
    pos_loss = tf.multiply(tf.multiply(-1.0, tf.log(pos_p)), label)
    neg_loss = tf.multiply(tf.multiply(-1.0, tf.log(neg_p)), (1.0 - label))


    pos_ratio = num_labels_neg / num_total
    neg_ratio = num_labels_pos / num_total
    sum_loss = pos_ratio * pos_loss + neg_ratio * neg_loss

    final_loss = tf.multiply((1.0 - final_p) ** 2, sum_loss)
    return tf.reduce_sum(final_loss)

def main():
    g = tf.Graph()
    # 构建一张计算图
    with g.as_default():
        # N W H C
        x = tf.placeholder(dtype=tf.float32, shape=[None, 224, 224, 3])
        f = tf.placeholder(dtype=tf.bool, shape=[])
        gtImg = tf.clip_by_value(tf.random_normal([5, 224, 224, conf.FIANL_CLASSES_NUM]), 0, 1)
        gtLb = tf.clip_by_value(tf.random_normal([5, conf.FIANL_CLASSES_NUM]), 0, 1)

        ## 如何外部调用
        logits_28, logits_56, logits_112, logits_224, logitsClass = LjchCNN(x,f)
        loss = lossFunc(logits_28, logits_56, logits_112, logits_224, logitsClass, gtImg, gtLb)
        initOpt = tf.global_variables_initializer()

    # 打开session开始计算
    with tf.Session(graph=g) as sess:
        sess.run([initOpt])

        vgg_var_list = tf.global_variables('resnet_v2_50')
        vgg_var_list = vgg_var_list[:-2]
        saver = tf.train.Saver(vgg_var_list)
        saver.restore(sess, 'model/resnet_v2_50.ckpt')

        in0 = np.clip(np.random.random([5, 224, 224, 3]), 0, 1)
        writer = tf.summary.FileWriter('log', sess.graph)
        vvx = tf.Variable(0)
        loss_ = sess.run(loss, feed_dict={x: in0,f:True})
        print(loss_)

if __name__ == '__main__':
    main()





LjchCNN = ResNet18Eyev1
backbone_name = 'resnet18' #'resnet18'



'''
如何使用其他的模型
    if vgg_loss:
        vgg_var_list = tf.global_variables('vgg_19')
        saver_vgg = tf.train.Saver(var_list=vgg_var_list)
        saver_vgg.restore(sess, 'vgg/models/vgg_19.ckpt')
        print('vgg params restored.')
var_list = list(set(tf.global_variables()) - set(vgg_var_list))
'''




'''


#
def main():
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        a = tf.zeros([3,224,224,3],dtype=tf.float32)
        zz = network(34,a,30,True)
        sess.run(tf.global_variables_initializer())
        zzz = sess.run(zz)



        # 创建Profiler实例作为记录、处理、显示数据的主体
        profiler = tf.profiler.Profiler(graph=sess.graph)

        # 设置trace_level，这样才能搜集到包含GPU硬件在内的最全统计数据
        #run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        # 创建RunMetadata实例，用于在每次sess.run时汇总统计数据
        # run_metadata = tf.RunMetadata()
        opts = tf.profiler.ProfileOptionBuilder.trainable_variables_parameter()
        param_stats = profiler.profile_name_scope(options=opts)
        # 总参数量
        print('总参数：', param_stats.total_parameters)
        # 各scope参数量
        z = 0
        for x in param_stats.children:
            z += x.total_parameters
            print(x.name, 'scope参数：', x.total_parameters)
        print('asdasd++++++++++++++++  ' + str(z))
        opts = tf.profiler.ProfileOptionBuilder.float_operation()
        float_stats = profiler.profile_operations(opts)
        # 总参数量
        print('总浮点运算数：', float_stats.total_float_ops)



        fvd = 34

    return 0


if __name__ == '__main__':
    main()





'''











