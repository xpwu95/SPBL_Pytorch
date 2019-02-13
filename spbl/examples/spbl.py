from __future__ import print_function, absolute_import
from reid.models import model_utils as mu
from reid.utils.data import data_process as dp
from reid import datasets
from reid import models
import numpy as np
import torch
import argparse
import os
from torch import nn
from collections import Counter

from reid.evaluation_metrics.metrics import classification_report_imbalanced
from reid.evaluation_metrics.metrics import classification_report_imbalanced_light
import scipy.io as scio
from sklearn.svm import SVC
import pickle
from examples.imblean.SMOTE.smote import SMOTE
from cfg import config

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"
def spaco(model_names,data,save_paths,iter_step=1,gamma=0.3,train_ratio=0.2):
    """
    self-paced balance learning model implementation based on Pytroch
    """
    assert iter_step >= 1
    assert len(model_names) == 1 and len(save_paths) == 1
    num_view = len(model_names)

    printflag = True

    add_ratios = config.add_ratios

    train_data,untrain_data = dp.split_dataset(data.train, train_ratio)#(data.trainval, train_ratio)
    data_dir = data.images_dir
    num_classes = data.num_trainval_ids

    res_model = mu.get_model_by_name(model_names[0],num_classes)
    res_model = nn.DataParallel(res_model).cuda()

    data_params = mu.get_params_by_name(model_names[0])

    res_model = mu.train_wxp(res_model,data.val,model_names[0],train_data,
                             data_dir,num_classes,config.epochs,weight=None)#weight)# - 5 * step) # modi

    clf = SVC(probability=False, decision_function_shape='ovr')

    # extract features
    torch.save(res_model, 'res50_198_temp.pkl')
    res_model.module.classifier = nn.LeakyReLU(0.1)
    features_train, labels_train = mu.extract_model_feature_ww(res_model, train_data, data_dir, data_params)
    if(printflag is True):
        features_test, labels_test = mu.extract_model_feature_ww(res_model, data.val, data_dir, data_params)
    features_untrain, _ = mu.extract_model_feature_ww(res_model, untrain_data, data_dir, data_params)

    features_paced = features_train.copy()
    labels_paced = labels_train.copy()
    sample_weights_paced = np.ones((len(labels_train),))

    clf.fit(features_paced, labels_paced,sample_weights_paced)
    sample_weights_paced = sample_weights_paced * 0.2

    if (printflag is True):
        pred_svm = clf.predict(features_test)

        pred_prob_svm = clf.decision_function(features_test)

        print('\n...svm...')
        print(classification_report_imbalanced_light(labels_test, pred_svm, pred_prob_svm, num_classes))

    pred_prob = clf.decision_function(features_untrain)
    pred_prob = pred_prob - np.max(pred_prob, axis=1).reshape((-1, 1))  # new
    pred_prob = np.exp(pred_prob)
    pred_prob = pred_prob / np.sum(pred_prob, axis=1).reshape((-1, 1))
    res_model = torch.load('res50_198_temp.pkl').module
    res_model = nn.DataParallel(res_model).cuda()

    weight = None
    weights = None
    weights_save = np.zeros((len(add_ratios),num_classes))
    weight_save = np.zeros((len(add_ratios), num_classes))
    subCurriculums = np.zeros((len(add_ratios), num_classes))


    for step in range(len(add_ratios)):

        # update v
        add_ratio = add_ratios[step]
        add_id, weights, subCurriculum = dp.sel_idx_wspl(pred_prob,untrain_data, add_ratio)
        #weights = weights / min(weights) # ensure min(weights) == 1.0
        '''
        if(weight is None):
            weight = weights.copy()
        else:
            weight = weight * 0.8 + weights * 0.2
        weight = weight / min(weight)'''
        weight = weights.copy()

        weight_save[step] = weight
        weights_save[step] = weights
        subCurriculums[step] = subCurriculum

        # update w
        new_train_data, _ = dp.update_train_untrain(
            add_id, train_data, untrain_data) # modi

        res_model = mu.train_wxp(res_model,data.val,model_names[0],new_train_data,
                               data_dir,num_classes,config.epochs,weight=None)#weight)# - 5 * step) # modi

        #extract features
        torch.save(res_model, 'res50_198_temp.pkl')
        res_model.module.classifier = nn.LeakyReLU(0.1)

        features_new_train, labels_new_train = mu.extract_model_feature_ww(res_model, new_train_data,data_dir, data_params)

        # curriculum reconstructing###
        #subCurriculum
        needsampled_indices = np.array([]).astype(int)#range(features_new_train.shape[0])
        sample_indices = np.array([]).astype(int)#range(features_new_train.shape[0])
        target_stats = Counter(labels_new_train)
        subCurriculum = subCurriculum.astype(int)
        for i in range(num_classes):
            target_class_indices = np.where(labels_new_train == i)[0]
            assert target_stats[i] == len(target_class_indices)
            if(subCurriculum[i] > 0):
                indices = range(target_stats[i])#target_class_indices.copy()
                indices = np.append(indices, np.random.randint(low=0, high=target_stats[i], size=subCurriculum[i]))
            elif(subCurriculum[i] == 0):
                indices = range(target_stats[i])#target_class_indices.copy()
            elif(subCurriculum[i] < 0):
                indices = np.array([]).astype(int)
                indices = np.append(indices, np.random.randint(low=0, high=target_stats[i], size=target_stats[i] + subCurriculum[i]))
            needsampled_indices = np.append(needsampled_indices, subCurriculum[i]+target_stats[i])
            sample_indices = np.append(sample_indices, target_class_indices[indices])
        features_new_train = features_new_train[sample_indices]
        labels_new_train = labels_new_train[sample_indices]

        '''
        RND_SEED = 42
        kind = 'regular'
        ratio = dict(zip(range(num_classes), needsampled_indices))
        smote = SMOTE(ratio=ratio,random_state=RND_SEED, k_neighbors=2, kind=kind)
        new_fe, new_la = smote.fit_sample(features_new_train, labels_new_train)
        features_new_train = new_fe.copy()# np.append(features_new_train, new_fe)
        labels_new_train = new_la.copy()#np.append(labels_new_train, new_la)
        '''
        #weight = weight[sample_indices]
        ##############################

        if (printflag is True or add_ratio > 0.9):
            features_test, labels_test = mu.extract_model_feature_ww(res_model, data.val,data_dir,data_params)
        features_untrain, _ = mu.extract_model_feature_ww(res_model, untrain_data,data_dir,data_params)

        sample_weights_paced_temp = np.zeros((len(labels_new_train),))
        for i, x in enumerate(labels_new_train):
            sample_weights_paced_temp[i] = weight[x]

        sample_weights_paced_temp_temp = np.concatenate((sample_weights_paced,sample_weights_paced_temp))

        features_paced = np.concatenate((features_paced,features_new_train))

        labels_paced = np.concatenate((labels_paced,labels_new_train))

        clf.fit(features_paced, labels_paced, sample_weights_paced_temp_temp)

        sample_weights_paced_temp = sample_weights_paced_temp * (add_ratio * (1 - train_ratio) + train_ratio)
        sample_weights_paced = np.concatenate((sample_weights_paced,sample_weights_paced_temp))

        if (printflag is True or add_ratio > 0.9):
            pred_svm = clf.predict(features_test)

            pred_prob_svm = clf.decision_function(features_test)

            print('\n...svm...')
            if add_ratio > 0.9:
                print(classification_report_imbalanced(labels_test, pred_svm, pred_prob_svm, num_classes))
            else:
                print(classification_report_imbalanced_light(labels_test, pred_svm, pred_prob_svm, num_classes))
        res_model = torch.load('res50_198_temp.pkl').module
        res_model = nn.DataParallel(res_model).cuda()

        # update proba
        pred_prob = clf.decision_function(features_untrain)
        pred_prob = pred_prob - np.max(pred_prob, axis=1).reshape((-1, 1))  # new
        pred_prob = np.exp(pred_prob)
        pred_prob = pred_prob / np.sum(pred_prob, axis=1).reshape((-1, 1))

        if (printflag is True or add_ratio > 0.9):
            pred_sm, lab_sm, score_sm = mu.pre_from_feature_ww(res_model,data.val,data_dir,data_params)

            print('\n...softmax...')
            if(add_ratio > 0.9):
                print(classification_report_imbalanced_light(lab_sm, pred_sm, score_sm, len(np.unique(lab_sm))))
            else:
                print(classification_report_imbalanced_light(lab_sm, pred_sm, score_sm, len(np.unique(lab_sm))))

def main(args):
    assert args.iter_step >= 1
    dataset_dir = os.path.join(args.data_dir, args.dataset)
    dataset = datasets.create(args.dataset, dataset_dir)
    model_names = [args.arch1]
    save_paths = [os.path.join(args.logs_dir, args.arch1)]
    spaco(model_names,dataset,save_paths,args.iter_step,
          args.gamma,args.train_ratio)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Self-paced balance learning')
    parser.add_argument('-d', '--dataset', type=str, default=config.dataset,
                        choices=datasets.names())
    parser.add_argument('-b', '--batch-size', type=int, default=config.batch_size)
    parser.add_argument('-a1', '--arch1', type=str, default=config.model,
                        choices=models.names())
    parser.add_argument('-i', '--iter-step', type=int, default=config.iter_step)
    parser.add_argument('-g', '--gamma', type=float, default=config.gamma)
    parser.add_argument('-r', '--train_ratio', type=float, default=config.train_ratio)
    working_dir = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default=os.path.join(working_dir,'data'))
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=os.path.join(working_dir,'logs'))
    main(parser.parse_args())
