import os
import random
from network import *
import datetime
import Utils.common as comm

'''################################################################################################
                          训练模型选择与预训练模型的选择

###################################################################################################
'''
## 调用模型的名称
FUNC_NAME = 'VGG16MASK'
## 损失函数的名称
LOSS_NAME = 'lossVGGMaskLoss'
## 辅助信息的显示
INFO = ''
## 模型训练文件的后缀
MODEL_Name = FUNC_NAME + '_' + str(datetime.datetime.now())[:13].replace('-','').replace(' ','')+INFO
## 调用模型的句柄
FUNC_HANDEL = eval(FUNC_NAME)
## 损失函数的句柄
LOSS_HANDLE = eval(LOSS_NAME)

## 是否提前载入预训练模型
LOAD_PRETRAIN = True
## 针对是否是预训练模型，设置不同的学习率
LR_PRETRAIN_DIFFERENT = False
## 预训练模型的学习率的减小
PRETRAIN_DECAY_RATE =  0
## 预训练模型的scope名称
PRETRAIN_SCOPE = 'resnet_v2_50'                 #'saliency'      #'VGG19CLS'  #'resnet_v2_50'
## 微调保留层数
RESERVE_LEVEL = 4
## 进行模型参数的偷偷保留
STEALTH_MODE_MODEL_ON = True


'''################################################################################################
                                    学习超参数

#################################################################################################
'''
## 学习率
LR = 1e-4#3
## 学习率下降间隔
LR_INTERVAL = 2000
## 学习率下降的比率
LR_DECAY_FACOTOR = 0.9
## L2 正则约束系数
WEIGHT_DECAY= 0#0.000000
## 批训练大小，训练
BATCH_SIZE = 8
## 批训练大小，测试
VALTEST_BATCHSIZE = 24

## 训练是按照Epoch，还是ITers
TRAIN_EPOCH_OR_ITERS = 'epoch'              # 'epoch','iter'
## 加载模型所用的线程数
NUM_WORKERS = 3
## 最大训练EPOCH次数
MAX_Epoch = 1000
## 最大训练迭代数目
MAX_ITERATIONS = 65000
## 冲量
#MOMENTUM = 0.9

## 梯度剪切
#GRADIENT_CLIP_THETA = 0.1
## 权重初始化参数
#WEIGHT_INIT_STDDEV_FACTOR = 0.1
#WEIGHT_INIT_MEAN_FACTOR = 0
'''################################################################################################
                                    路径配置

#################################################################################################
'''
# 路径配置
_PROJECT_BASEPATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASETS_PATH = os.path.join(_PROJECT_BASEPATH,'Datasets','Neijing/blendDS')
DATASETS_TRAIN_PATH = os.path.join(DATASETS_PATH,'train')
DATASETS_VAL_PATH = os.path.join(DATASETS_PATH,'val')
TRAIN_FN = os.path.join(DATASETS_PATH, 'Train.tfrecord')
VAL_FN = os.path.join(DATASETS_PATH, 'Val.tfrecord')
PRETRAINED_VGG19 = os.path.join(_PROJECT_BASEPATH,'/pretrainedMod/vgg_19.ckpt')
PRETRAINED_Resnet50 = os.path.join(_PROJECT_BASEPATH,'/pretrainedMod/resnet_v2_50.ckpt')
PRETRAINED_VGG19NPY = os.path.join(_PROJECT_BASEPATH,'/pretrainedMod/vgg16.npy')
MODEL_PATH = os.path.join(_PROJECT_BASEPATH,'Model','Neijing_%s','model') % MODEL_Name
LOG_PATH = os.path.join(_PROJECT_BASEPATH,'Model','Neijing_%s','log') % MODEL_Name
STEALTH_MODE_MODEL_PATH = os.path.join(_PROJECT_BASEPATH,'Model','Neijing_%s','snapshots') % MODEL_Name


# 网络结构



'''################################################################################################
                                    网络模型参数

#################################################################################################
'''
FIANL_CLASSES_NUM = 5
SAMPLE_H = 300
SAMPLE_W = 300
STD_INPUT_H = 224
STD_INPUT_W = 224



'''################################################################################################
                                         其他参数

#################################################################################################
'''
PRINT_INTERVAL = 50
SAVE_INTERVAL = 1000
SUMMARY_INTERVAL = 50
VALIDATION_INTERVAL = 500
STEALTH_INTERVAL = 1000


GPU_FLAG = True
GPUS = 0
SEED = random.randint(1, 900000)




