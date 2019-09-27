## image IO
import imageio
import skimage

# tf
import  tensorflow as tf
import  tensorboard
import matplotlib.pyplot as plt
from skimage import transform,data




# Traditional
import os
import numpy as np
import time
import cv2 as cv
from datetime import datetime
import math
from tqdm import tqdm
import skimage
# Program
import Config as conf
import Utils.common as tools
from tqdm import tqdm

# network

from tensorflow.contrib.slim import nets
from network import LjchCNN,lossFunc


def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def randomResize(img):
    shapeFactor = tf.random_uniform([2], minval=0.7, maxval=1.3)
    shape = tf.cast(img.shape,tf.float32)[:-1]
    targetShape = tf.cast(tf.clip_by_value(shapeFactor * shape,230,23333),tf.int32)
    newImg = tf.image.resize_images(img,targetShape,align_corners=True)
    return newImg


def adjustImage(img):
    img = tf.image.random_brightness(img, max_delta=23. / 255.)
    img = tf.image.random_contrast(img, lower=0.79, upper=1.3)
    return img

def data_augmentation(image,label,training=True):
    if training:
        image_label = tf.concat([image,label],axis = -1)

        # 放缩 随机旋转角度
        k = tf.random_uniform([], maxval=1.0)
        maybe_flipped = tf.cond(tf.greater(k , 0.8),lambda:randomResize(image_label),
                              lambda: tf.py_func(pyfunc_RandomRotate, [image_label], tf.float32))
        # 转90°
        k = tf.random_uniform([], maxval=1.0)
        maybe_flipped = tf.cond(tf.greater(k , 0.5),lambda:tf.image.rot90(maybe_flipped),lambda: maybe_flipped )

        # 左右上下
        maybe_flipped = tf.image.random_flip_left_right(maybe_flipped)
        maybe_flipped = tf.image.random_flip_up_down(maybe_flipped)

        # 随机剪切
        maybe_flipped = tf.random_crop(maybe_flipped,size=[conf.STD_INPUT_H, conf.STD_INPUT_W, image_label.get_shape()[-1]])

        # 拆分
        image = maybe_flipped[:, :, :3]
        mask = maybe_flipped[:, :, 3:]


        # 调节图像的亮度与对比度
        k = tf.random_uniform([], maxval=1.0)
        image = tf.cond(tf.greater(k, 0.7),lambda:adjustImage(image),lambda:image)

        return image, mask




def readAndDecode(filename, augmentation=True):

    '''
    直接读取数据，不加入预处理(制作数据集的时候已经预处理完毕)
    labelFat.shape          (300, 300, 5)
    img.shape               (300, 300, 3)
    clsLabel.shape          (5,)
    '''

    filename_queue = tf.train.string_input_producer([filename], num_epochs=None)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(
        serialized_example,
        features={

            'image': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.string),
            'clsLabel': tf.FixedLenFeature([], tf.string)
        }
    )
    label = features['label']
    image = features['image']
    clsLabel = features['clsLabel']

    # decode images and normalization for input

    image = tf.decode_raw(image, tf.uint8)
    image = tf.reshape(image, [conf.SAMPLE_H, conf.SAMPLE_W, 3])
    image = tf.cast(image, tf.float32)


    # label images
    label = tf.decode_raw(label, tf.float32)
    label_shape = tf.stack([conf.SAMPLE_H, conf.SAMPLE_W, conf.FIANL_CLASSES_NUM])
    label = tf.reshape(label, label_shape)
    # label = tf.cast(tf.greater(label, 0.5), tf.float32)

    # clsLabel
    clsLabel = tf.decode_raw(clsLabel, tf.float32)
    clsLabel = tf.reshape(clsLabel, [1,1,conf.FIANL_CLASSES_NUM])
    clsLabel = tf.cast(clsLabel, tf.float32)
    if augmentation:
        image, label = data_augmentation(image,label)
    else:
        pass

    return image, label, clsLabel




def makeRecord():
    dsTrainPath = conf.DATASETS_TRAIN_PATH
    dsValPath = conf.DATASETS_VAL_PATH

    trainFn = conf.TRAIN_FN
    valFn = conf.VAL_FN

    writerTrain = tf.python_io.TFRecordWriter(trainFn)
    writerVal = tf.python_io.TFRecordWriter(valFn)
    # build Train
    ## target Tensor and shape:
    trainImg = os.path.join(dsTrainPath,'img')
    trainLb  = os.path.join(dsTrainPath,'label')
    fnSet = os.listdir(trainImg)
    print('开始准备训练集...')
    lenn = len(fnSet)
    for  idx in tqdm(range(lenn)):
        fn = fnSet[idx]
        imgPath = os.path.join(trainImg,fn)
        lbPaht = os.path.join(trainLb,fn)

        img = skimage.io.imread(imgPath)
        label = skimage.io.imread(lbPaht)
        # resize image
        img = skimage.transform.resize(img, (conf.SAMPLE_H, conf.SAMPLE_W))
        img = skimage.img_as_ubyte(img)
        # resize convert label
        label = skimage.transform.resize(label, (conf.SAMPLE_H, conf.SAMPLE_W))
        label = label > 0.5;label = label.astype(np.float32)
        labelFat = labelConverter(label,5)
        unique_label = np.unique(label)
        # 如何定义类别标签
        clsLabel = 1
        if len(unique_label) == 1 and unique_label[0] == 0:
            clsLabel = 0
        clsLabel = clsLabelConverter(clsLabel, conf.FIANL_CLASSES_NUM)

        # recording
        example = tf.train.Example(features=tf.train.Features(feature={
            'label': _bytes_feature(labelFat.tostring()),
            'image': _bytes_feature(img.tostring()),
            'clsLabel': _bytes_feature(clsLabel.tostring())
        }))
        writerTrain.write(example.SerializeToString())
    writerTrain.close()
    print('训练集数据准备完毕')

    #build Val
    ## target Tensor and shape:
    valImg = os.path.join(dsValPath,'img')
    valLb  = os.path.join(dsValPath,'label')
    fnSet = os.listdir(valImg)
    print('开始准备验证集...')
    lenn = len(fnSet)
    for  idx in tqdm(range(lenn)):
        fn = fnSet[idx]
        imgPath = os.path.join(valImg,fn)
        lbPaht = os.path.join(valLb,fn)
        # load images
        img = skimage.io.imread(imgPath)
        label = skimage.io.imread(lbPaht)
        # resize image
        img = skimage.transform.resize(img, (conf.SAMPLE_H, conf.SAMPLE_W))
        img = skimage.img_as_ubyte(img)
        # resize convert label
        label = skimage.transform.resize(label, (conf.SAMPLE_H, conf.SAMPLE_W))
        label = label > 0.5;
        label = label.astype(np.float32)
        labelFat = labelConverter(label, 5)
        # clsLabel and converter
        unique_label = np.unique(label)
        clsLabel = 1
        if len(unique_label) == 1 and unique_label[0] == 0:
            clsLabel = 0
        # clsLabel Converter
        clsLabel = clsLabelConverter(clsLabel, conf.FIANL_CLASSES_NUM)

        # recording
        example = tf.train.Example(features=tf.train.Features(feature={
            'label': _bytes_feature(labelFat.tostring()),
            'image': _bytes_feature(img.tostring()),
            'clsLabel': _bytes_feature(clsLabel.tostring())
        }))
        writerVal.write(example.SerializeToString())
    writerVal.close()
    print('验证集数据准备完毕')


def clsLabelConverter(label,numCls):
    arr = np.arange(numCls)
    res = np.array(arr == label)
    res = res.astype(np.float32)
    res = res.reshape([1,1,numCls])
    return  res


def labelConverter(label,numCls):
    assert  numCls > 1
    h,w = label.shape
    newLabel = np.zeros([h,w,numCls],dtype=np.float32)
    for i in range(numCls):
        chann = (label == i)
        chann = chann.astype(np.float32)
        newLabel[:,:,i] =chann
    return newLabel



def pyfunc_RandomRotate(imNMask):
    angle = np.random.uniform(low=-20.0, high=20.0)
    im = imNMask[:,:,:3]
    mask = imNMask[:,:,3:]
    imNew = transform.rotate(im, angle, resize=True)
    maskNew = transform.rotate(mask, angle, resize=True)

    targetEdgeLen = 300 / (np.sqrt(2) * np.sin(deg2arc(np.abs(angle) + 45)))
    delta = np.maximum(int((imNew.shape[0] - targetEdgeLen) / 2),1)

    tarImg = imNew[delta:-delta, delta:-delta, :]
    tarMask = maskNew[delta:-delta, delta:-delta]
    tarImg.astype(np.float32)

    tarMask = tarMask > 0.5
    tarMask = tarMask.astype(np.float32)

    if len(tarMask.shape) == 2:
        tarMask = np.expand_dims(tarMask, axis=-1)

    res = np.concatenate([tarImg,tarMask],axis=-1)

    return res.astype(np.float32)



def main():
    #makeRecord
    x = np.random.rand(300,300,8)
    while True:
        v = pyfunc_RandomRotate(x,)



def arc2deg(arc):
    coef = 180 / np.pi
    return coef * arc

def deg2arc(deg):
    coef = np.pi / 180
    return coef * deg

if __name__ == '__main__':
    main()


