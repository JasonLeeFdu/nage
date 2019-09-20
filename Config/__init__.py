import os
import random

# For Training
MODEL_Name = 'Resnet18MapScore_20190920'



_PROJECT_BASEPATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASETS_PATH = os.path.join(_PROJECT_BASEPATH,'Datasets','Neijing/blendDS')
DATASETS_TRAIN_PATH = os.path.join(DATASETS_PATH,'train')
DATASETS_VAL_PATH = os.path.join(DATASETS_PATH,'val')
TRAIN_FN = os.path.join(DATASETS_PATH, 'Train.tfrecord')
VAL_FN = os.path.join(DATASETS_PATH, 'Val.tfrecord')
PRETRAINED_VGG19 = os.path.join(_PROJECT_BASEPATH,'/pretrainedMod/vgg_19.ckpt')
PRETRAINED_Resnet50 = os.path.join(_PROJECT_BASEPATH,'/pretrainedMod/resnet_v2_50.ckpt')



# 网络结构
FIANL_CLASSES_NUM = 5
SAMPLE_H = 300
SAMPLE_W = 300
STD_INPUT_H = 224
STD_INPUT_W = 224


MODEL_PATH = os.path.join(_PROJECT_BASEPATH,'Model','Neijing_%s','model') % MODEL_Name
LOG_PATH = os.path.join(_PROJECT_BASEPATH,'Model','Neijing_%s','log') % MODEL_Name


NUM_WORKERS = 3
MAX_Epoch = 101
MAX_ITERATIONS = 2000000
PRINT_INTERVAL = 50
SAVE_INTERVAL = 1000
SUMMARY_INTERVAL = 50
VALIDATION_INTERVAL = 100
GPU_FLAG = True
GPUS = 0
SEED = random.randint(1, 900000)
LR_INTERVAL = 50000
WEIGHT_DECAY= 3.3e-4# 1e-3 #0.000000
MOMENTUM = 0.9
PRELOAD = False




## Hyper parameters concerning with training performance and Gradient Deminish or ex
## GRADIENT_CLIP = 0.1                     　# small
LR = 1e-3#3                                　# small 0.0005
BATCH_SIZE = 8                              # X
VALTEST_BATCHSIZE = 24





#######
WEIGHT_INIT_STDDEV_FACTOR = 1.3                # big
WEIGHT_INIT_MEAN_FACTOR = 0
SUMMARY_SCALAR_FIX  = 3e-3
GRADIENT_CLIP_THETA = 0.1



## 模型测试
# TEST_MODEL_PATH = '/home/winston/workSpace/PycharmProjects/Foundation/AutoEncoder/testModels'




