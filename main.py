## image IO
import imageio
import skimage

# tf
import  tensorflow as tf

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
import recordMaker as rm
from network import *
import matplotlib.pyplot as plot
import tensorflow.contrib.slim as slim



'''
输入归一化的问题                            => done
L2 norm of weight-weight decay           => done
L2 gamma beta                            => done
学习率的问题                               => done
保证batchnorm中的gamma beta 问题           => done (还是采用了gradient apply)


precision recall IOU 等专用医用数据
医用数据代码对接
医用独立分类接口

tensorboard                              => done
断点续传                                  => done
确保trian test 一致问题                    => done
实现train val                            => done

'''


def trainClsModel():
    # 目录
    tools.securePath(conf.LOG_PATH)
    tools.securePath(conf.MODEL_PATH)
    with tf.Session() as sess:
        ## summarize 的名字
        trainWriter = tf.summary.FileWriter(os.path.join(conf.LOG_PATH, 'train'), sess.graph)
        testWriter = tf.summary.FileWriter(os.path.join(conf.LOG_PATH, 'val'), sess.graph)

        # 输入数据 input data
        trainSwitch = tf.placeholder(dtype=tf.bool)
        coord = tf.train.Coordinator()
        imageTrain, labelTrain, clsLabelTrain = rm.readAndDecode(conf.TRAIN_FN)
        imageVal, labelVal, clsLabelVal = rm.readAndDecode(conf.VAL_FN)
        imageTrainB, labelTrainB, clsLabelTrainB = tf.train.shuffle_batch([imageTrain, labelTrain, clsLabelTrain],
                                                                          batch_size=conf.BATCH_SIZE,
                                                                          capacity=512,
                                                                          min_after_dequeue=256, num_threads=3)
        imageValB, labelValB, clsLabelValB = tf.train.shuffle_batch([imageVal, labelVal, clsLabelVal],
                                                                    batch_size=conf.VALTEST_BATCHSIZE,
                                                                    capacity=512,
                                                                    min_after_dequeue=256, num_threads=3)

        # 用于读取数据的多个线程(队列读取线程)的协调，否则会在线程暂停的时候报错
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        # 输入管道的架设(训练验证分支)
        image = tf.placeholder_with_default(input=imageTrainB, shape=[None, 224, 224, 3])
        label = tf.placeholder_with_default(input=labelTrainB, shape=[None, 224, 224, conf.FIANL_CLASSES_NUM])
        clsLabel = tf.placeholder_with_default(input=clsLabelTrainB, shape=[None, 1, 1, conf.FIANL_CLASSES_NUM])

        # 构建网络
        logitsClass, predCls = conf.FUNC_HANDEL(image, trainSwitch)
        lossFunc = conf.LOSS_HANDLE( logitsClass, clsLabel)
        loadPretrainedResnetVGG19(sess)
        # loadPretrainedResnetVGG19(sess)

        # 正则化项

        weights_var = tf.trainable_variables()
        weights_var_withoutNorm = [x for x in weights_var if
                                   ((x.name.find('gamma') == -1) and (x.name.find('beta') == -1))]
        l2NormLoss = conf.WEIGHT_DECAY * tf.add_n([tf.nn.l2_loss(v) for v in weights_var_withoutNorm]) / len(
            weights_var_withoutNorm)

        # 统一loss
        loss = lossFunc + l2NormLoss

        # 记录哪一些重要的节点
        tf.summary.image("input", image)
        tf.summary.image("label", tf.expand_dims(label[:, :, :, 1], axis=-1))
        tf.summary.scalar("Loss(networkAndL2)", loss)

        # 全局训练次数
        globalStep = tf.Variable(0, name='global_step', trainable=False)

        # 不同层设置不同的学习率
        learning_rate = tf.train.exponential_decay(conf.LR, tf.train.get_or_create_global_step(), conf.LR_INTERVAL, 0.1,
                                                   staircase=True)
        if backbone_name == 'resnet18':
            optimizer = tf.train.AdamOptimizer(learning_rate)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)  # 为了保证bn 顺利工作
            with tf.control_dependencies(update_ops):
                trainOpts = optimizer.minimize(loss, tf.train.get_or_create_global_step())
                optLr2 = optimizer._lr
        else:
            weights_var_all = tf.trainable_variables()
            # weights_var_all_withoutNorm = [x for x in weights_var_all if ((x.name.find('gamma') == -1) and (x.name.find('beta') == -1))]
            weights_var_resnet = tf.trainable_variables(scope='vgg_19')
            # weights_var_resnet_withoutNorm = [x for x in weights_var_resnet if ((x.name.find('gamma') == -1) and (x.name.find('beta') == -1))]
            weights_var_FPNBridge = [x for x in weights_var_all if x not in weights_var_resnet]
            # weights_var_FPNBridge_withoutNorm = [x for x in weights_var_all_withoutNorm if x not in weights_var_resnet_withoutNorm]
            var_list1 = weights_var_resnet  # 模型参数
            var_list2 = weights_var_FPNBridge
            opt1 = tf.train.AdamOptimizer(learning_rate / 20)
            optLr1 = opt1._lr
            opt2 = tf.train.AdamOptimizer(learning_rate)
            optLr2 = opt2._lr
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)  # 为了保证bn 顺利工作
            with tf.control_dependencies(update_ops):
                grads = tf.gradients(loss, var_list1 + var_list2)
                grads1 = grads[:len(var_list1)]
                grads2 = grads[len(var_list1):]
                train_op1 = opt1.apply_gradients(zip(grads1, var_list1), global_step=globalStep)  # only once
                train_op2 = opt2.apply_gradients(zip(grads2, var_list2))
                trainOpts = tf.group(train_op1, train_op2)

        # 全局变量赋值
        initOpt = tf.global_variables_initializer()
        sess.run(initOpt)

        # 载入之前训练的模型
        saver = tf.train.Saver(max_to_keep=10)
        ckpt = tf.train.get_checkpoint_state(conf.MODEL_PATH)
        start_it = 1
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_str = ckpt.model_checkpoint_path
            start_it = int(ckpt_str[ckpt_str.find('-') + 1:]) + 1
            print('Continue training at Iter %d' % start_it)
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print('No training model found, start from iter 1')

        # 训练循环
        print('开始训练')
        AvgFreq = 0
        Avgloss = 0
        mergedSummOpt = tf.summary.merge_all()

        for iter1 in np.arange(start_it, conf.MAX_ITERATIONS):
            startTime = time.time()  # 统计
            _, lossData, lr2, summ = sess.run([trainOpts, loss, optLr2, mergedSummOpt],
                                              feed_dict={trainSwitch: True})  # 训练，注意sess.run 的第一个参数是一个fetch list
            endTime = time.time()
            # 更新统计变量
            AvgFreq += endTime - startTime
            Avgloss += lossData

            ## 显示
            if (iter1 % conf.PRINT_INTERVAL == 0) and (iter1 != 0):  # 显示  gs
                AvgFreq = (conf.PRINT_INTERVAL * conf.BATCH_SIZE) / AvgFreq
                Avgloss = Avgloss / conf.PRINT_INTERVAL
                # format_str = '%s: Iters %d, average loss(255) = %.7f, average frequency = %.3f(HZ), LR = %.10f'
                format_str = '%s: Iters %d, average loss(255) = %.7f, average frequency = %.3f(HZ), LR = %.10f'
                if iter1 % 1000 == 0:
                    print(format_str % (datetime.now(), iter1, Avgloss, AvgFreq, lr2), end='')
                else:
                    print(format_str % (datetime.now(), iter1, Avgloss, AvgFreq, lr2))
                AvgFreq = 0
                Avgloss = 0
            ## 保存
            if (iter1 % conf.SAVE_INTERVAL == 0) and (iter1 != 0):  # 显示
                saver.save(sess, os.path.join(conf.MODEL_PATH, 'model.ckpt'), global_step=iter1)
                print(' ... Iter %d model2 saved! ' % iter1)
            ## summary
            if (iter1 % conf.SUMMARY_INTERVAL == 0) and (iter1 != 0):  # 显示
                trainWriter.add_summary(summ, iter1)
            ## test and summary

            if (iter1 % conf.VALIDATION_INTERVAL == 0) and (iter1 != 0):  # 显示
                trainWriter.add_summary(summ, iter1)
                ima, lab, cls = sess.run([imageValB, labelValB, clsLabelValB])

                lossData, summ = sess.run([loss, mergedSummOpt],
                                          feed_dict={trainSwitch: False, image: ima, label: lab,
                                                     clsLabel: cls})  # 训练，注意sess.run 的第一个参数是一个fetch list
                testWriter.add_summary(summ, iter1)

                # print('测试完毕')
        # close the queue reading threads// used for multi-queue readers
        coord.request_stop()
        coord.join(threads)



def trainTogether():
    # 目录
    tools.securePath(conf.LOG_PATH)
    tools.securePath(conf.MODEL_PATH)

    with tf.Session() as sess:
        ## summarize 的名字
        trainWriter = tf.summary.FileWriter(os.path.join(conf.LOG_PATH,'train'), sess.graph)
        testWriter = tf.summary.FileWriter(os.path.join(conf.LOG_PATH,'val'), sess.graph)

        # 输入数据 input data
        trainSwitch = tf.placeholder(dtype=tf.bool)
        coord = tf.train.Coordinator()
        imageTrain, labelTrain, clsLabelTrain = rm.readAndDecode(conf.TRAIN_FN)
        imageVal, labelVal, clsLabelVal = rm.readAndDecode(conf.VAL_FN)
        imageTrainB, labelTrainB, clsLabelTrainB = tf.train.shuffle_batch([imageTrain, labelTrain, clsLabelTrain], batch_size=conf.BATCH_SIZE,
                                                                    capacity=512,
                                                                    min_after_dequeue=256,num_threads=3)
        imageValB, labelValB, clsLabelValB = tf.train.shuffle_batch([imageVal, labelVal, clsLabelVal], batch_size=conf.VALTEST_BATCHSIZE,
                                                                    capacity=512,
                                                                    min_after_dequeue=256,num_threads=3)

        # 用于读取数据的多个线程(队列读取线程)的协调，否则会在线程暂停的时候报错
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        # 输入管道的架设(训练验证分支)
        image = tf.placeholder_with_default(input=imageTrainB,shape=[None,224,224,3])
        label = tf.placeholder_with_default(input=labelTrainB,shape=[None,224,224,conf.FIANL_CLASSES_NUM])
        clsLabel = tf.placeholder_with_default(input=clsLabelTrainB,shape=[None,1,1,conf.FIANL_CLASSES_NUM])

        # 构建网络
        logits_28, logits_56, logits_112, logits_224, logitMask, logitsClass, predFlat, predCls, predVis = conf.FUNC_HANDEL(image,trainSwitch)
        networkLoss = conf.LOSS_HANDLE(logits_28, logits_56, logits_112, logits_224, logitMask, logitsClass, label, clsLabel, predVis)
        # loadPretrainedResnetVGG19(sess)
        # loadPretrainedResnetVGG19(sess)

        # 正则化项
        if backbone_name != 'resnet18':  #正常流程
            weights_var = tf.trainable_variables()
            weights_var_withoutNorm = [x for x in weights_var if
                                       ((x.name.find('gamma') == -1) and (x.name.find('beta') == -1))]
            l2NormLoss = conf.WEIGHT_DECAY * tf.add_n([tf.nn.l2_loss(v) for v in weights_var_withoutNorm]) / len(
                weights_var_withoutNorm)

        else:
            # 单独处理vgg
            weights_var = tf.trainable_variables()
            weights_var_withoutNorm = [x for x in weights_var if
                                       ((x.name.find('gamma') == -1) and (x.name.find('beta') == -1))]
            weights_var_FC = [x for x in weights_var if (x.name.find('dense') != -1)]
            weights_var_WithoutFC = [x for x in weights_var_withoutNorm if x not in weights_var_FC]
            l2NormLoss1 = 3 * conf.WEIGHT_DECAY * tf.add_n([tf.nn.l2_loss(v) for v in weights_var_FC]) / len(
                weights_var_withoutNorm)
            l2NormLoss2 = conf.WEIGHT_DECAY * tf.add_n([tf.nn.l2_loss(v) for v in weights_var_WithoutFC]) / len(
                weights_var_withoutNorm)
            l2NormLoss = l2NormLoss1 + l2NormLoss2



        # 统一loss
        loss = networkLoss + l2NormLoss

        # 记录哪一些重要的节点
        tf.summary.image("input",image)
        tf.summary.image("label",tf.expand_dims(label[:,:,:,1],axis=-1))
        tf.summary.image("predVis(C1)",tf.expand_dims(predVis,axis=-1))

        tf.summary.scalar("Loss(networkAndL2)", loss)



        # 全局训练次数
        globalStep = tf.Variable(0, name='global_step', trainable=False)


        # 不同层设置不同的学习率
        learning_rate = tf.train.exponential_decay(conf.LR, tf.train.get_or_create_global_step(), conf.LR_INTERVAL, 0.1, staircase=True)
        if backbone_name == 'resnet18':
            optimizer = tf.train.AdamOptimizer(learning_rate)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)  # 为了保证bn 顺利工作
            with tf.control_dependencies(update_ops):
                trainOpts = optimizer.minimize(loss,tf.train.get_or_create_global_step())
                optLr2 = optimizer._lr
        else:
            weights_var_all = tf.trainable_variables()
            # weights_var_all_withoutNorm = [x for x in weights_var_all if ((x.name.find('gamma') == -1) and (x.name.find('beta') == -1))]
            weights_var_resnet = tf.trainable_variables(scope='vgg_19')
            #weights_var_resnet_withoutNorm = [x for x in weights_var_resnet if ((x.name.find('gamma') == -1) and (x.name.find('beta') == -1))]
            weights_var_FPNBridge = [x for x in weights_var_all if x not in weights_var_resnet]
            #weights_var_FPNBridge_withoutNorm = [x for x in weights_var_all_withoutNorm if x not in weights_var_resnet_withoutNorm]
            var_list1 = weights_var_resnet                          # 模型参数
            var_list2 = weights_var_FPNBridge
            opt1 = tf.train.AdamOptimizer(learning_rate / 20)
            optLr1 = opt1._lr
            opt2 = tf.train.AdamOptimizer(learning_rate)
            optLr2 = opt2._lr
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) # 为了保证bn 顺利工作
            with tf.control_dependencies(update_ops):
                grads = tf.gradients(loss, var_list1 + var_list2)
                grads1 = grads[:len(var_list1)]
                grads2 = grads[len(var_list1):]
                train_op1 = opt1.apply_gradients(zip(grads1, var_list1),global_step=globalStep) # only once
                train_op2 = opt2.apply_gradients(zip(grads2, var_list2))
                trainOpts = tf.group(train_op1, train_op2)

        # 全局变量赋值
        initOpt = tf.global_variables_initializer()
        sess.run(initOpt)

        # 载入之前训练的模型
        saver = tf.train.Saver(max_to_keep=10)
        ckpt = tf.train.get_checkpoint_state(conf.MODEL_PATH)
        start_it = 1
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_str = ckpt.model_checkpoint_path
            start_it = int(ckpt_str[ckpt_str.find('-') + 1:]) + 1
            print('Continue training at Iter %d' % start_it)
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print('No training model found, start from iter 1')

        # 训练循环
        print('开始训练')
        AvgFreq = 0
        Avgloss = 0
        mergedSummOpt = tf.summary.merge_all()


        for iter1 in np.arange(start_it,conf.MAX_ITERATIONS):
            startTime = time.time()  # 统计
            _,lossData,lr2,summ = sess.run([trainOpts, loss, optLr2,mergedSummOpt],feed_dict={trainSwitch:True})              #训练，注意sess.run 的第一个参数是一个fetch list
            endTime = time.time()
            # 更新统计变量
            AvgFreq += endTime - startTime
            Avgloss += lossData

            ## 显示
            if (iter1 % conf.PRINT_INTERVAL == 0) and (iter1 != 0):                          # 显示  gs
                AvgFreq = (conf.PRINT_INTERVAL * conf.BATCH_SIZE) / AvgFreq
                Avgloss = Avgloss / conf.PRINT_INTERVAL
                # format_str = '%s: Iters %d, average loss(255) = %.7f, average frequency = %.3f(HZ), LR = %.10f'
                format_str = '%s: Iters %d, average loss(255) = %.7f, average frequency = %.3f(HZ), LR = %.10f'
                if iter1 % 1000 == 0:
                    print(format_str % (datetime.now(), iter1, Avgloss, AvgFreq,lr2),end='')
                else:
                    print(format_str % (datetime.now(), iter1, Avgloss, AvgFreq,lr2))
                AvgFreq = 0
                Avgloss = 0
            ## 保存
            if (iter1 % conf.SAVE_INTERVAL == 0) and (iter1 != 0):  # 显示
                saver.save(sess, os.path.join(conf.MODEL_PATH,'model.ckpt'), global_step=iter1)
                print(' ... Iter %d model2 saved! ' % iter1)
            ## summary
            if (iter1 % conf.SUMMARY_INTERVAL == 0) and (iter1 != 0):  # 显示
                trainWriter.add_summary(summ, iter1)
            ## test and summary

            if (iter1 % conf.VALIDATION_INTERVAL == 0) and (iter1 != 0):  # 显示
                trainWriter.add_summary(summ, iter1)
                ima,lab,cls = sess.run([imageValB, labelValB, clsLabelValB])

                lossData, summ = sess.run([loss, mergedSummOpt],
                                                  feed_dict={trainSwitch: False ,image:ima,label:lab,clsLabel:cls})  # 训练，注意sess.run 的第一个参数是一个fetch list
                testWriter.add_summary(summ,iter1)

                # print('测试完毕')
        # close the queue reading threads// used for multi-queue readers
        coord.request_stop()
        coord.join(threads)


def main():
    trainClsModel()



if __name__ == '__main__':
    main()




























































































































































