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

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"
def spaco(model_names,data,save_paths,iter_step=1,gamma=0.3,train_ratio=0.2):
    """
    self-paced co-training model implementation based on Pytroch
    params:
    model_names: model names for spaco, such as ['resnet50','densenet121']
    data: dataset for spaco model
    save_pathts: save paths for two models
    iter_step: iteration round for spaco
    gamma: spaco hyperparameter
    train_ratio: initiate training dataset ratio
    """
    assert iter_step >= 1
    assert len(model_names) == 1 and len(save_paths) == 1
    num_view = len(model_names)

    printflag = True#False#

    train_ratio = 0.2#0.2
    #add_ratio = 0.5
    # add_ratios = [0.5,1.0,1.0,1.0,1.0]#
    add_ratios = [0.5, 0.75, 0.9, 1.0, 1.0, 1.0]#[0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 0.99, 1.0, 1.0]#
    # add_ratios = [0.29, 0.42, 0.57, 0.71, 0.86, 1.0]#[0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 0.99, 1.0, 1.0]#
    #add_step = 0.5


    train_data,untrain_data = dp.split_dataset(data.train, train_ratio)#(data.trainval, train_ratio)
    data_dir = data.images_dir
    num_classes = data.num_trainval_ids
    ###########
    # initiate classifier to get preidctions
    ###########

    res_model = mu.get_model_by_name(model_names[0],num_classes)
    # res_model = torch.load('res50_cv1_epoch29_0.628.pkl').module
    # print(res_model)
    res_model = nn.DataParallel(res_model).cuda()

    '''
    data_params = mu.get_params_by_name(model_names[0])
    pred_sm, lab_sm, score_sm = mu.pre_from_feature_ww(res_model,data.val,data_dir,data_params)
    print(classification_report_imbalanced_light(lab_sm, pred_sm, score_sm, len(np.unique(lab_sm))))

    clf = SVC(probability=False, decision_function_shape='ovr')

    torch.save(res_model, 'res50_198_temp.pkl')
    res_model.module.classifier = nn.LeakyReLU(0.1)
    features_train, labels_train = mu.extract_model_feature_ww(res_model, data.train, data_dir, data_params)
    features_test, labels_test = mu.extract_model_feature_ww(res_model, data.val, data_dir, data_params)

    clf.fit(features_train, labels_train)
    pred_svm = clf.predict(features_test)
    pred_prob_svm = clf.decision_function(features_test)
    print('\n...svm...')
    print(classification_report_imbalanced_light(labels_test, pred_svm, pred_prob_svm, num_classes))
    res_model = torch.load('res50_198_temp.pkl').module
    res_model = nn.DataParallel(res_model).cuda()

    # input()

    fname = 'res50_cv1_epoch29_0.628_features_train.npy'
    lname = 'res50_cv1_epoch29_0.628_labels_train.npy'
    np.save(fname, features_train)
    np.save(lname, labels_train)
    fname = 'res50_cv1_epoch29_0.628_features_test.npy'
    lname = 'res50_cv1_epoch29_0.628_labels_test.npy'
    np.save(fname, features_test)
    np.save(lname, labels_test)

    input()
    '''
    '''
    #test acne level4
    # res_model = mu.train_wxp(res_model,data.val,model_names[0],data.train,
    #                          data_dir,num_classes,100,weight=None)#weight)# - 5 * step) # modi
    # torch.save(res_model, 'res50_acne_level_4_temp.pkl')
    data_params = mu.get_params_by_name(model_names[0])
    pred_sm, lab_sm, score_sm = mu.pre_from_feature_ww(res_model,data.val,data_dir,data_params)
    print(classification_report_imbalanced(lab_sm, pred_sm, score_sm, len(np.unique(lab_sm))))
    input()

    res_model = mu.train_wxp(res_model,data.val,model_names[0],train_data,
                             data_dir,num_classes,50,weight=None)#weight)# - 5 * step) # modi

    clf = SVC(probability=False, decision_function_shape='ovr')

    #test acne_level4
    torch.save(res_model, 'res50_198_temp.pkl')
    res_model.module.classifier = nn.LeakyReLU(0.1)
    data_params = mu.get_params_by_name(model_names[0])
    features_train, labels_train = mu.extract_model_feature_ww(res_model, data.train, data_dir, data_params)
    features_test, labels_test = mu.extract_model_feature_ww(res_model, data.val, data_dir, data_params)
    clf.fit(features_train, labels_train)

    pred_svm = clf.predict(features_test)
    pred_prob_svm = clf.decision_function(features_test)
    print('\n...svm...')
    print(classification_report_imbalanced(labels_test, pred_svm, pred_prob_svm, num_classes))
    res_model = torch.load('res50_198_temp.pkl').module
    res_model = nn.DataParallel(res_model).cuda()
    input()
    '''

    '''
    features_train = np.load('res50_ori_198_epoch18_0.593_features_train.npy')
    features_test = np.load('res50_ori_198_epoch18_0.593_features_test.npy')
    labels_train = np.load('res50_ori_198_epoch18_0.593_labels_train.npy')
    labels_test = np.load('res50_ori_198_epoch18_0.593_labels_test.npy')

    clf.fit(features_train, labels_train)
    pred_svm = clf.predict(features_test)
    pred_prob_svm = clf.decision_function(features_test)
    pred_prob_svm = pred_prob_svm - np.max(pred_prob_svm, axis=1).reshape((-1, 1))  # new
    pred_prob_svm = np.exp(pred_prob_svm)
    pred_prob_svm = pred_prob_svm / np.sum(pred_prob_svm, axis=1).reshape((-1, 1))

    ppname_mat_test = 'res50_ori_198_epoch18_0.593_preprob_test.mat'
    pname_mat_test = 'res50_ori_198_epoch18_0.593_pre_test.mat'
    scio.savemat(ppname_mat_test, {'PreProb_test': pred_prob_svm})
    scio.savemat(pname_mat_test, {'Pre_test': pred_svm})
    input()'''
    '''
    #save score mat
    data_params = mu.get_params_by_name(model_names[0])
    pred_, des_y, score = mu.pre_from_feature_ww(res_model, data.val,data_dir,data_params)
    ppname_mat_test = 'res50_ori_198_epoch18_0.593_preprob_test.mat'
    pname_mat_test = 'res50_ori_198_epoch18_0.593_pre_test.mat'
    scio.savemat(ppname_mat_test, {'PreProb_test': score})
    scio.savemat(pname_mat_test, {'Pre_test': pred_})

    #save mat
    features_train = np.load('res50_ori_198_epoch18_0.593_features_train.npy')
    features_test = np.load('res50_ori_198_epoch18_0.593_features_test.npy')
    labels_train = np.load('res50_ori_198_epoch18_0.593_labels_train.npy')
    labels_test = np.load('res50_ori_198_epoch18_0.593_labels_test.npy')
    fname_mat_train = 'res50_ori_198_epoch18_0.593_features_train.mat'
    fname_mat_test = 'res50_ori_198_epoch18_0.593_features_test.mat'
    lname_mat_train = 'res50_ori_198_epoch18_0.593_labels_train.mat'
    lname_mat_test = 'res50_ori_198_epoch18_0.593_labels_test.mat'
    scio.savemat(fname_mat_train, {'Fe_train': features_train})
    scio.savemat(fname_mat_test, {'Fe_test': features_test})
    scio.savemat(lname_mat_train, {'La_train': labels_train})
    scio.savemat(lname_mat_test, {'La_test': labels_test})

    #save fetures
    data_params = mu.get_params_by_name(model_names[0])
    res_model.module.classifier = nn.LeakyReLU(0.1)
    features_, labels_ = mu.extract_model_feature_ww(res_model, data.train,data_dir,data_params)
    fname = 'res50_ori_260_epoch64_0.551_features_train.npy'
    lname = 'res50_ori_260_epoch64_0.551_labels_train.npy'
    np.save(fname, features_)
    np.save(lname, labels_)
    '''

    '''
    # test resnet-50
    res_model = mu.train_wxp(res_model,data.val,model_names[0],data.train,
                             data_dir,num_classes,50)# - 5 * step) # modi
    data_params = mu.get_params_by_name(model_names[0])
    pred_, des_y, score = mu.pre_from_feature_ww(res_model, data.val,data_dir,data_params)
    numt = np.where(pred_ == des_y)[0]
    print('test resnet-50')
    print(numt)
    print(numt.size)
    print('-----')
    print(classification_report_imbalanced(des_y, pred_, score, num_classes))
    input()
    '''

    #'''
    data_params = mu.get_params_by_name(model_names[0])

    #pred_prob = mu.train_predict_w(res_model,
    #        model_names[0],train_data,untrain_data,num_classes,data_dir,epochs=50,valdata=data.val)
    res_model = mu.train_wxp(res_model,data.val,model_names[0],train_data,
                             data_dir,num_classes,50,weight=None)#weight)# - 5 * step) # modi

    clf = SVC(probability=False, decision_function_shape='ovr')

    # extract features
    torch.save(res_model, 'res50_198_temp.pkl')
    res_model.module.classifier = nn.LeakyReLU(0.1)
    features_train, labels_train = mu.extract_model_feature_ww(res_model, train_data, data_dir, data_params)
    if(printflag is True):
        features_test, labels_test = mu.extract_model_feature_ww(res_model, data.val, data_dir, data_params)
    features_untrain, _ = mu.extract_model_feature_ww(res_model, untrain_data, data_dir, data_params)
    # np.save('features_train_mit67_pace%d' % 0, features_train)
    # np.save('labels_train_mit67_pace%d' % 0, labels_train)
    # np.save('weights_train_mit67_pace%d' % 0, np.ones((len(labels_train),)))
    features_paced = features_train.copy()
    labels_paced = labels_train.copy()
    sample_weights_paced = np.ones((len(labels_train),))
    #clf.fit(features_train, labels_train)
    clf.fit(features_paced, labels_paced,sample_weights_paced)
    sample_weights_paced = sample_weights_paced * 0.2

    if (printflag is True):
        pred_svm = clf.predict(features_test)
        # pred_prob_svm  = clf.predict_proba(features_test)
        pred_prob_svm = clf.decision_function(features_test)
        # pred_prob_svm = pred_prob_svm - np.max(pred_prob_svm, axis=1).reshape((-1, 1))  # new
        # pred_prob_svm = np.exp(pred_prob_svm)
        # pred_prob_svm = pred_prob_svm / np.sum(pred_prob_svm, axis=1).reshape((-1, 1))
        print('\n...svm...')
        print(classification_report_imbalanced_light(labels_test, pred_svm, pred_prob_svm, num_classes))
    # pred_prob = clf.predict_proba(features_untrain)
    pred_prob = clf.decision_function(features_untrain)
    pred_prob = pred_prob - np.max(pred_prob, axis=1).reshape((-1, 1))  # new
    pred_prob = np.exp(pred_prob)
    pred_prob = pred_prob / np.sum(pred_prob, axis=1).reshape((-1, 1))
    res_model = torch.load('res50_198_temp.pkl').module
    res_model = nn.DataParallel(res_model).cuda()
    #'''

    '''
    pred_sm, lab_sm, score_sm = mu.pre_from_feature_ww(res_model, data.val, data_dir, data_params)
    print('\n...softmax...')
    print(classification_report_imbalanced_light(lab_sm, pred_sm, score_sm, len(np.unique(lab_sm))))
    '''

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

        # weight = np.ones((num_classes,)) # weight = 1 only for curriculum constructing

        # update w
        new_train_data, _ = dp.update_train_untrain(
            add_id, train_data, untrain_data) # modi

        res_model = mu.train_wxp(res_model,data.val,model_names[0],new_train_data,
                               data_dir,num_classes,50,weight=None)#weight)# - 5 * step) # modi

        #extract features
        torch.save(res_model, 'res50_198_temp.pkl')
        res_model.module.classifier = nn.LeakyReLU(0.1)

        features_new_train, labels_new_train = mu.extract_model_feature_ww(res_model, new_train_data,data_dir, data_params)
        # np.save('features_train_mit67_pace%d_noCR' % (step + 1), features_new_train)
        # np.save('labels_train_mit67_pace%d_noCR' % (step + 1), labels_new_train)
        # np.save('weights_train_mit67_pace%d_noCR' % (step + 1), weight)
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
        #nweight = weight[sample_indices]
        ##############################

        if (printflag is True or add_ratio > 0.9):
            features_test, labels_test = mu.extract_model_feature_ww(res_model, data.val,data_dir,data_params)
            # np.save('features_test_mit67_pace%d' % (step + 1), features_test)
            # np.save('labels_test_mit67_pace%d' % (step + 1), labels_test)
        features_untrain, _ = mu.extract_model_feature_ww(res_model, untrain_data,data_dir,data_params)

        sample_weights_paced_temp = np.zeros((len(labels_new_train),))
        for i, x in enumerate(labels_new_train):
            sample_weights_paced_temp[i] = weight[x]
        # np.save('features_train_mit67_pace%d' % (step + 1), features_new_train)
        # np.save('labels_train_mit67_pace%d' % (step + 1), labels_new_train)
        # np.save('weights_train_mit67_pace%d' % (step + 1), sample_weights_paced_temp)
        sample_weights_paced_temp_temp = np.concatenate((sample_weights_paced,sample_weights_paced_temp))
        # features_paced = features_new_train.copy()#
        features_paced = np.concatenate((features_paced,features_new_train))
        # labels_paced = labels_new_train.copy()#
        labels_paced = np.concatenate((labels_paced,labels_new_train))

        #assert len(features_paced) == len(labels_paced)
        #assert len(features_paced) == len(sample_weights_paced_temp_temp)
        #clf.fit(features_new_train,labels_new_train)
        clf.fit(features_paced, labels_paced, sample_weights_paced_temp_temp)

        sample_weights_paced_temp = sample_weights_paced_temp * (add_ratio * (1 - train_ratio) + train_ratio)
        sample_weights_paced = np.concatenate((sample_weights_paced,sample_weights_paced_temp))

        if (printflag is True or add_ratio > 0.9):
            pred_svm = clf.predict(features_test)
            # pred_prob_svm  = clf.predict_proba(features_test)
            pred_prob_svm = clf.decision_function(features_test)
            # pred_prob_svm = pred_prob_svm - np.max(pred_prob_svm, axis=1).reshape((-1, 1))  # new
            # pred_prob_svm = np.exp(pred_prob_svm)
            # pred_prob_svm = pred_prob_svm / np.sum(pred_prob_svm, axis=1).reshape((-1, 1))
            # np.save('pred_svm198_step%d'%(step),pred_svm)
            # np.save('predprob_svm198_step%d' % (step), pred_prob_svm)
            print('\n...svm...')
            if add_ratio > 0.9:
                print(classification_report_imbalanced(labels_test, pred_svm, pred_prob_svm, num_classes))
            else:
                print(classification_report_imbalanced_light(labels_test, pred_svm, pred_prob_svm, num_classes))
        res_model = torch.load('res50_198_temp.pkl').module
        res_model = nn.DataParallel(res_model).cuda()

        # update proba
        # pred_prob = clf.predict_proba(features_untrain)
        pred_prob = clf.decision_function(features_untrain)
        pred_prob = pred_prob - np.max(pred_prob, axis=1).reshape((-1, 1))  # new
        pred_prob = np.exp(pred_prob)
        pred_prob = pred_prob / np.sum(pred_prob, axis=1).reshape((-1, 1))
        # pred_prob = mu.predict_prob(
        #    res_model,untrain_data,data_dir,data_params)
        # pred_prob = pred_prob * weight.T

        if (printflag is True or add_ratio > 0.9):
            pred_sm, lab_sm, score_sm = mu.pre_from_feature_ww(res_model,data.val,data_dir,data_params)
            #test max weighted-output
            #pred_sm = np.argmax(score_sm * weights, axis=1)
            #score_sm = score_sm * weight.T
            #pred_sm = np.argmax(score_sm, axis=1)
            print('\n...softmax...')
            if(add_ratio > 0.9):
                print(classification_report_imbalanced_light(lab_sm, pred_sm, score_sm, len(np.unique(lab_sm))))
            else:
                print(classification_report_imbalanced_light(lab_sm, pred_sm, score_sm, len(np.unique(lab_sm))))
        #torch.save(res_model, 'res50_SPL_198_step%d_%.3f.pkl' % (step, 1.0 * len(np.where(pred_sm==lab_sm)[0])/lab_sm.shape[0]))


    # np.save('weight_save', weight_save)
    # np.save('weights_save', weights_save)
    # np.save('subCurriculums', subCurriculums)


def main(args):
    assert args.iter_step >= 1
    dataset_dir = os.path.join(args.data_dir, args.dataset)
    dataset = datasets.create(args.dataset, dataset_dir)
    model_names = [args.arch1]#, args.arch2]
    save_paths = [os.path.join(args.logs_dir, args.arch1)]#,
                  #os.path.join(args.logs_dir, args.arch2)]
    spaco(model_names,dataset,save_paths,args.iter_step,
          args.gamma,args.train_ratio)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Self-paced cotraining Reid')
    #parser.add_argument('-d', '--dataset', type=str, default='market1501',
    parser.add_argument('-d', '--dataset', type=str, default='mcr',#'mnist',#'indoor', #'sd-198',#sd-198',
                        choices=datasets.names())
    parser.add_argument('-b', '--batch-size', type=int, default=64)
    parser.add_argument('-a1', '--arch1', type=str, default='resnet50',
                        choices=models.names())
    #parser.add_argument('-a2', '--arch2', type=str, default='densenet121',
    #                    choices=models.names())
    parser.add_argument('-i', '--iter-step', type=int, default=5)
    parser.add_argument('-g', '--gamma', type=float, default=0.3)
    parser.add_argument('-r', '--train_ratio', type=float, default=0.2)
    working_dir = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default=os.path.join(working_dir,'data'))
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=os.path.join(working_dir,'logs'))
    main(parser.parse_args())
