import torch
from torch import nn
from reid import models
from reid.trainers import Trainer
from reid.evaluators import extract_features, extract_features_w, Evaluator
from reid.dist_metric import DistanceMetric
from reid.utils.data import data_process as dp
import numpy as np
from collections import OrderedDict

from torch.autograd import Variable
#from ..evaluation_metrics import accuracy
from ..utils import to_torch
from reid.evaluation_metrics.metrics import classification_report_imbalanced_light
_FEATURE_NUM = 128
_DROPOUT = 0.3
_PARAMS_FACTORY = {
    'resnet':
        {'height':224,
            'width':224},
    'inception':
        {'height':128,
            'width':64},
    'inception_v3':
        {'height':299,
            'width':299},
    'densenet':
        {'height':224,
            'width':224},
    'vgg':
        {'height':224,
            'width':224},
    'mnist':
        {'height':28,
            'width':28}
}


def get_model_by_name(model_name,num_classes):
    """
    create model given the model_name and number of classes
    """
    if 'resnet' in model_name:
        model = models.create(model_name,num_features=4096,#128,
                              dropout=0.3,num_classes=num_classes)
    elif 'inception' in model_name:
        model = models.create(model_name,num_features=128,
                              dropout=0.3,num_classes=num_classes)
    elif 'densenet' in model_name:
        model = models.create(model_name,num_features=128,
                              dropout=0.3, num_classes=num_classes)
    elif 'vgg' in model_name:
        model = models.create(model_name,num_features=4096,#128,
                              dropout=0.3, num_classes=num_classes)
    elif 'mnist' in model_name:
        model = models.create(model_name,num_features=4096,#128,
                              dropout=0.3, num_classes=num_classes)
    else:
        raise ValueError('wrong model name, no such model!')
    return model


def get_params_by_name(model_name):
    """
    get model Parameters given the model_name
    """
    params = {}
    for k,v in _PARAMS_FACTORY.items():
        if k in model_name:
            params = v
    if not params:
        raise ValueError('wrong model name, no params!')
    params['batch_size'] = 64
    params['workers'] = 2 # ?
    return params


def train_model(model,dataloader,epochs=50):
    """
    train model given the dataloader the criterion,
    stop when epochs are reached
    params:
        model: model for training
        dataloader: training data
        epochs: training epochs
        criterion
    """
    if hasattr(model.module, 'base'):
        base_param_ids = set(map(id, model.module.base.parameters()))
        new_params = [p for p in model.parameters() if
                      id(p) not in base_param_ids]
        param_groups = [
            {'params': model.module.base.parameters(), 'lr_mult': 0.1},
            {'params': new_params, 'lr_mult': 1.0}]
    else:
        param_groups = model.parameters()
    optimizer = torch.optim.SGD(param_groups, lr=0.1,
                                momentum=0.9,
                                weight_decay=5e-4,
                                nesterov=True)

    def adjust_lr(epoch):
        step_size = 40
        lr = 0.1 * (0.1 ** (epoch // step_size)) # 0.1 * 0.1^(epoch divide step_size)
        #lr *= 0.1 # 0.0001 ?
        for g in optimizer.param_groups:
            g['lr'] = lr * g.get('lr_mult', 1)
    criterion = nn.CrossEntropyLoss().cuda()
    trainer = Trainer(model,criterion)
    ac_now = 0.0
    ac_max = 0.1
    losses_avg = 0.0
    losses_avg_c = 0

    for epoch in range(epochs):
        adjust_lr(epoch)
        losses_avg = trainer.train(epoch, dataloader, optimizer)
        '''
        if losses_avg < 0.003:
            print('terminate: losses_avg < 0.003')
            return
        elif losses_avg < 0.008 and losses_avg > 0.007:
            losses_avg_c = losses_avg_c + 1
            if(losses_avg_c >= 10):
                print('terminate: 0.007 < losses_avg < 0.008')
                return
        '''
    '''
        if((epoch + 1) % 1 == 0):
            pred_, lab_, score = pre_from_feature_ww(model,testdata,data_dir,data_params)
            numt = np.where(pred_ == lab_)[0]
            ac_now = numt.size / len(lab_)
            print('test data.val %.3f' % (ac_now))
            #print(classification_report_imbalanced_light(lab_, pred_, score, len(np.unique(lab_))))
            #print(numt)
            if(ac_now > ac_max):
                ac_max = ac_now
                torch.save(model, 'res50_SPL_198_maxtemp.pkl')

    model = torch.load('res50_SPL_198_maxtemp.pkl').module
    model = nn.DataParallel(model).cuda()
    return model
    '''

def train(model_name,train_data,data_dir,num_classes,epochs=50):
    model = get_model_by_name(model_name,num_classes)
    model = nn.DataParallel(model).cuda()
    data_params = get_params_by_name(model_name)
    dataloader = dp.get_dataloader(
        train_data,data_dir,training=True,**data_params)
    train_model(model,dataloader,epochs=epochs)
    return model


def get_feature(model,data,data_dir,params):
    dataloader = dp.get_dataloader(data,data_dir,**params)
    features,_ = extract_features(model,dataloader)
    return features


def predict_prob(model,data,data_dir,params):
    features = get_feature(model,data,data_dir,params)
    logits = np.array([logit.numpy() for logit in features.values()])
    logits = logits - np.max(logits,axis=1).reshape((-1,1)) # new
    exp_logits = np.exp(logits)
    predict_prob = exp_logits / np.sum(exp_logits,axis=1).reshape((-1,1))
    assert len(logits) == len(predict_prob)
    return predict_prob


def train_predict(model_name,train_data,untrain_data,num_classes,data_dir):
    model = train(model_name,train_data,data_dir,num_classes)
    data_params = get_params_by_name(model_name)
    pred_prob = predict_prob(model,untrain_data,data_dir,data_params)
    return pred_prob


def get_clusters(model,data_loader,num_classes):
    features, labels = extract_features(model, data_loader)
    class_features = OrderedDict(list)
    for k,v in labels.items():
        class_features[v].append(features[k])
    clusters = [np.mean(class_features[i],axis=0)
                for i in range(num_classes)]
    clusters = torch.from_numpy(np.array(clusters))
    return torch.autograd.Variable(clusters)


def evaluate(model,dataset,params,metric=None):
    val = dataset.val
    '''query,gallery = dataset.query,dataset.gallery
    dataloader = dp.get_dataloader(
        list(set(dataset.query) | set(dataset.gallery)),
        dataset.images_dir,**params)
    metric = DistanceMetric(algorithm='euclidean')
    metric.train(model,dataloader)
    evaluator = Evaluator(model)
    evaluator.evaluate(dataloader,query,gallery,metric)'''

    dataloader = dp.get_dataloader(val,dataset.images_dir,**params)

    '''features,_ = extract_features(model,dataloader)
    pred_y = np.argmax(sum(pred_probs),axis=1)'''

    #model = nn.DataParallel(model).cuda()
    #criterion = nn.CrossEntropyLoss().cuda()
    #trainer = Trainer(model,criterion)

    correct = 0
    total = 0
    for inputs in dataloader: # enumerate(dataloader):
        imgs, _, pids, _ = inputs
        inputs = [Variable(imgs)]
        targets = pids.cuda(async=True)#Variable(pids.cuda())#pids.cuda()

        outputs = model(*inputs)

        total += targets.data.size(0)
        correct += accuracy(outputs.data, targets)#.data)

        '''_, pred = torch.max(outputs.data, 1)
        #pred = pred.t()
        outputs, targets = to_torch(outputs), to_torch(targets)
        total += targets.size(0)
        correct += (pred == targets).sum
        #correct += pred.eq(target.view(1, -1).expand_as(pred))
        #prec, = accuracy(outputs.data, targets.data)
        #prec = prec[0]'''
    print('%f\n' % (correct / total))


def accuracy(output, target, topk=(1,)):
    #output, target = to_torch(output), to_torch(target)
    maxk = max(topk)
    #batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    #ret = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(dim=0, keepdim=True)
        #ret.append(correct_k.mul_(1. / batch_size))
    return correct_k #ret

def train_model_w(model,dataloader,testdata,data_dir,data_params,epochs=50,weights=None):
    """
    train model given the dataloader the criterion,
    stop when epochs are reached
    params:
        model: model for training
        dataloader: training data
        epochs: training epochs
        criterion
    """
    if hasattr(model.module, 'base'):
        base_param_ids = set(map(id, model.module.base.parameters()))
        new_params = [p for p in model.parameters() if
                      id(p) not in base_param_ids]
        param_groups = [
            {'params': model.module.base.parameters(), 'lr_mult': 0.1},
            {'params': new_params, 'lr_mult': 1.0}]
    else:
        param_groups = model.parameters()
    # optimizer = torch.optim.Adam(params=param_groups, lr=0.1, weight_decay=5e-4,amsgrad=True)
    optimizer = torch.optim.SGD(param_groups, lr=0.01,
                                momentum=0.9,
                                weight_decay=5e-4,
                                nesterov=True)

    def adjust_lr(epoch):
        step_size =40#20#40
        #0-150 0.1 150-250 0.01 250-350 0.001
        #resume 0.01
        # if epoch == 100:
        #     step_size = 50
        # if epoch < 60: expand = 0
        # elif epoch < 80: expand = 1
        # else: expand = 2
        lr = 0.01 * (0.1 ** (epoch // step_size)) # 0.1 * 0.1^(epoch divide step_size)
        #if(epoch >= 70):
        #    lr *= 0.1
        for g in optimizer.param_groups:
            g['lr'] = lr * g.get('lr_mult', 1)
    if(weights is not None):
        criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(weights).float()).cuda()
    else:
        criterion = nn.CrossEntropyLoss().cuda()
    trainer = Trainer(model,criterion)
    ac_now = 0.0
    ac_max = 0.1
    losses_avg = 0.0
    losses_avg_c = 0
    for epoch in range(epochs):
        adjust_lr(epoch)
        losses_avg = trainer.train(epoch, dataloader, optimizer)
        '''
        if losses_avg <= 0.004:
            print('terminate: losses_avg < 0.003')
            return model
        elif losses_avg < 0.008 and losses_avg > 0.006:
            losses_avg_c = losses_avg_c + 1
            if(losses_avg_c >= 3):
                print('terminate: 0.007 < losses_avg < 0.008')
                return model
        '''
        if (epoch + 1) % 1 == 0 and epoch >= 9:
            pred_, lab_, score = pre_from_feature_ww(model,testdata,data_dir,data_params)
            numt = np.where(pred_ == lab_)[0]
            ac_now = numt.size / float(len(lab_))
            print('test data.val %.3f' % (ac_now))
            # print(classification_report_imbalanced_light(lab_, pred_, score, len(np.unique(lab_))))
            #print(numt)
            if (ac_now >= ac_max):
                ac_max = ac_now
                torch.save(model, 'res50_sd260_cross_val_maxtemp.pkl')
            # torch.save(model, 'res50_cv3_epoch%d_%.3f.pkl' % (epoch, ac_now))

    model = torch.load('res50_sd260_cross_val_maxtemp.pkl').module
    model = nn.DataParallel(model).cuda()
    return model

def train_w(model,model_name,train_data,data_dir,num_classes,epochs=50):
    data_params = get_params_by_name(model_name)
    dataloader = dp.get_dataloader(
        train_data,data_dir,training=True,**data_params)
    train_model(model,dataloader,epochs=epochs)
    return model

def train_wxp(model,testdata,model_name,train_data,data_dir,num_classes,epochs=50,weight=None):
    data_params = get_params_by_name(model_name)
    dataloader = dp.get_dataloader(
        train_data,data_dir,training=True,**data_params)
    #train_model(model,dataloader,epochs=epochs)
    model = train_model_w(model,dataloader,testdata,data_dir,data_params,epochs=epochs,weights=weight)
    return model

def train_predict_w(model,model_name,train_data,untrain_data,num_classes,data_dir,epochs=50):
    model = train_w(model,model_name,train_data,data_dir,num_classes,epochs=epochs)
    data_params = get_params_by_name(model_name)
    pred_prob = predict_prob(model,untrain_data,data_dir,data_params)
    return pred_prob

def pre_from_feature_w(model,data,data_dir,params):
    dataloader = dp.get_dataloader(data,data_dir,**params)
    features,labels = extract_features_w(model,dataloader)
    features = np.array([logit.numpy() for logit in features.values()])
    pred = np.argmax(features,axis=1)
    labels = np.array([logit for logit in labels.values()])
    return pred, labels

def extract_model_feature_ww(model,data,data_dir,params):
    dataloader = dp.get_dataloader(data,data_dir,**params)
    features,labels = extract_features_w(model,dataloader)
    features = np.array([logit.numpy() for logit in features.values()])
    labels = np.array([logit for logit in labels.values()])

    return features, labels

def pre_from_feature_ww(model,data,data_dir,params):
    dataloader = dp.get_dataloader(data,data_dir,**params)
    features,labels = extract_features_w(model,dataloader)
    features = np.array([logit.numpy() for logit in features.values()])
    pred = np.argmax(features,axis=1)
    labels = np.array([logit for logit in labels.values()])

    features = features - np.max(features,axis=1).reshape((-1,1)) # new
    features = np.exp(features)
    features = features / np.sum(features,axis=1).reshape((-1,1))
    return pred, labels, features

def extract_label(data,data_dir,params):

    labels = OrderedDict()
    dataloader = dp.get_dataloader(data,data_dir,**params)

    for i, (imgs, fnames, pids, _) in enumerate(dataloader):

        for fname, pid in zip(fnames, pids):
            labels[fname] = pid

    labels = np.array([logit for logit in labels.values()])
    return labels
