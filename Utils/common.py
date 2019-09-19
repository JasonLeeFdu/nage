import os
import Config as conf
import pickle
import io
from thop import  profile


def securePath(path):
    if not os.path.exists(path):
        os.mkdir(path)
    return

def secureSoftLink(src,dst):
    ## src dst
    if not os.path.exists(dst):
        os.symlink(src,dst)
    return



def saveCheckpoint(netModel, epoch, iterr, glbiter, fnCore='model'):
    ##net_state = netModel.state_dict()
    res = dict()
    ##res['NetState'] = net_state
    res['NetState'] = netModel
    res['Epoch'] = epoch
    res['Iter'] = iterr
    res['GlobalIter'] = glbiter
    fn = fnCore + '_' + str(epoch) + '_' + str(iterr) + '.mdl'
    pfn = os.path.join(conf.MODEL_PATH, fn)
    pfnFile = io.open(pfn, mode='wb')
    pickle.dump(res, pfnFile)


def loadSpecificCheckpointNetState1(epoch, iterr, fnCore='model'):
    fn = fnCore + '_' + str(epoch) + '_' + str(iterr) + '.mdl'
    pfn = os.path.join(conf.MODEL_PATH, fn)
    res = pickle.load(pfn)
    net_state = res['NetState']
    globalIter = res['GlobalIter']
    return net_state, globalIter


def loadLatestCheckpoint(fnCore='model'):
    # return net_status epoch iterr
    modelPath = conf.MODEL_PATH
    candidateCpSet = os.listdir(modelPath)
    candidateCpSet = [x for x in candidateCpSet if x.startswith(fnCore) and x.endswith('.mdl')]
    if len(candidateCpSet) == 0:
        return None, 0, 0, 0
    ref = [x.split('.')[0] for x in candidateCpSet]
    ref1 = [x.split('_')[1] for x in ref]
    ref2 = [x.split('_')[2] for x in ref]
    factor = 10 ** len(sorted(ref2, key=lambda k: len(k), reverse=True)[0])
    ref1 = [int(x) for x in ref1]
    ref2 = [int(x) for x in ref2]
    reff = list(zip(ref1, ref2))
    reff = [x[0] * factor + x[1] for x in reff]
    idx = reff.index(max(reff))
    latestCpFn = candidateCpSet[idx]
    latestCpFnFIO = io.open(os.path.join(modelPath, latestCpFn), 'rb')
    res = pickle.load(latestCpFnFIO)
    return res['NetState'], res['Epoch'], res['Iter'], res['GlobalIter']

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def loadModelsByPath(path):
    file = io.open(path, 'rb')
    res = pickle.load(file)
    return res['NetState'], res['Epoch'], res['Iter'], res['GlobalIter']

def loadStandAloneModelsByPath(path):
    file = io.open(path, 'rb')
    res = pickle.load(file)
    return res['NET']

def netDict2Net(dirName,net):
    print('Converting model paramdict to standalone network.')
    fnSet = os.listdir(dirName)
    fnSet = [x for x in fnSet if x[-4:]=='.mdl']
    dstDir = 'ConvertedStandaloneMethods'
    securePath(os.path.join(dirName,dstDir))
    for fn in fnSet:
        latestCpFnFIO = io.open(os.path.join(dirName, fn), 'rb')
        res = pickle.load(latestCpFnFIO)
        net.load_state_dict(res['NetState'])
        res['NET'] = net
        pfn = os.path.join(dirName,dstDir,fn)
        pfnFile = io.open(pfn, mode='wb')
        pickle.dump(res, pfnFile)
        print('Done: %s'% pfn)

def sizeofNet(net,inputs):
    # GFLOPs = 10 ^ 9 FLOPs  flops, params = sizeofNet(net,[torch.rand(1,3,128,128).cuda()])

    flops,params = profile(net,inputs=inputs)
    return flops,params


