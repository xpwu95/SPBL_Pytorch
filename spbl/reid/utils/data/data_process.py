import numpy as np
from reid.utils.data import transforms as T
from torch.utils.data import DataLoader
from reid.utils.data.preprocessor import Preprocessor
import torchvision

import matlab.engine
from PIL import Image


class RandomRotation(object):
    """
    https://github.com/pytorch/vision/tree/master/torchvision/transforms
    Rotate the image by angle.
    Args:
        degrees (sequence or float or int): Range of degrees to select from.
            If degrees is a number instead of sequence like (min, max), the range of degrees
            will be (-degrees, +degrees).
        resample ({PIL.Image.NEAREST, PIL.Image.BILINEAR, PIL.Image.BICUBIC}, optional):
            An optional resampling filter.
            See http://pillow.readthedocs.io/en/3.4.x/handbook/concepts.html#filters
            If omitted, or if the image has mode "1" or "P", it is set to PIL.Image.NEAREST.
        expand (bool, optional): Optional expansion flag.
            If true, expands the output to make it large enough to hold the entire rotated image.
            If false or omitted, make the output image the same size as the input image.
            Note that the expand flag assumes rotation around the center and no translation.
        center (2-tuple, optional): Optional center of rotation.
            Origin is the upper left corner.
            Default is the center of the image.
    """

    def __init__(self, degrees, resample=False, expand=False, center=None):
        # if isinstance(degrees, numbers.Number):
        #     if degrees < 0:
        #         raise ValueError("If degrees is a single number, it must be positive.")
        self.degrees = (-degrees, degrees)
        # else:
        #     if len(degrees) != 2:
        #         raise ValueError("If degrees is a sequence, it must be of len 2.")
        #     self.degrees = degrees

        self.resample = resample
        self.expand = expand
        self.center = center

    @staticmethod
    def get_params(degrees):
        """Get parameters for ``rotate`` for a random rotation.
        Returns:
            sequence: params to be passed to ``rotate`` for random rotation.
        """
        angle = np.random.uniform(degrees[0], degrees[1])

        return angle

    def __call__(self, img):
        """
            img (PIL Image): Image to be rotated.
        Returns:
            PIL Image: Rotated image.
        """

        def rotate(img, angle, resample=False, expand=False, center=None):
            """Rotate the image by angle and then (optionally) translate it by (n_columns, n_rows)
            Args:
            img (PIL Image): PIL Image to be rotated.
            angle ({float, int}): In degrees degrees counter clockwise order.
            resample ({PIL.Image.NEAREST, PIL.Image.BILINEAR, PIL.Image.BICUBIC}, optional):
            An optional resampling filter.
            See http://pillow.readthedocs.io/en/3.4.x/handbook/concepts.html#filters
            If omitted, or if the image has mode "1" or "P", it is set to PIL.Image.NEAREST.
            expand (bool, optional): Optional expansion flag.
            If true, expands the output image to make it large enough to hold the entire rotated image.
            If false or omitted, make the output image the same size as the input image.
            Note that the expand flag assumes rotation around the center and no translation.
            center (2-tuple, optional): Optional center of rotation.
            Origin is the upper left corner.
            Default is the center of the image.
            """

            return img.rotate(angle, resample, expand, center)

        angle = self.get_params(self.degrees)

        return rotate(img, angle, self.resample, self.expand, self.center)


class RandomShift(object):
    def __init__(self, shift):
        self.shift = shift

    @staticmethod
    def get_params(shift):
        """Get parameters for ``rotate`` for a random rotation.
        Returns:
            sequence: params to be passed to ``rotate`` for random rotation.
        """
        hshift, vshift = np.random.uniform(-shift, shift, size=2)

        return hshift, vshift

    def __call__(self, img):
        hshift, vshift = self.get_params(self.shift)

        return img.transform(img.size, Image.AFFINE, (1, 0, hshift, 0, 1, vshift), resample=Image.BICUBIC, fill=1)

def get_dataloader(dataset,data_dir,
                   training=False, height=256,
                   width=128, batch_size=64, workers=1):
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])#(0.1307,), (0.3081,)
    # normalizer = T.Normalize(mean=[0.1307, 0.1307, 0.1307],
    #                          std=[0.3081, 0.3081, 0.3081])#, (0.3081,)

    if training:
        transformer = T.Compose([
            T.RectScale(256, 256),
            #T.RandomSizedRectCrop(height, width),
            torchvision.transforms.RandomCrop(224),
            #T.RandomHorizontalFlip()
            torchvision.transforms.RandomHorizontalFlip(),
            # RandomRotation(degrees=20),
            # RandomShift(3),
            T.ToTensor(),
            normalizer,
        ])
    else:
        transformer = T.Compose([
            T.RectScale(224, 224),
            #torchvision.transforms.CenterCrop(320),
            T.ToTensor(),
            normalizer,
        ])
    data_loader = DataLoader(
        Preprocessor(dataset, root=data_dir,
                     transform=transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=training, pin_memory=True, drop_last=training)
    return data_loader

def update_train_untrain(sel_idx,train_data,untrain_data):
    assert len(train_data[0]) == len(untrain_data[0])
    #add_data = [(untrain_data[i][0],int(pred_y[i]),untrain_data[i][2])
    add_data = [untrain_data[i]
                for i,flag in enumerate(sel_idx) if flag]
    data1 = [untrain_data[i]
             for i,flag in enumerate(sel_idx) if not flag]
    data2 = train_data + add_data
    return data2, data1


def sel_idx(score,untrain_data,add_ratio=0.5):
    y = np.array([label for _,label,_ in untrain_data])
    add_indices = np.zeros(score.shape[0])
    clss = np.unique(y)
    assert score.shape[1] == len(clss)
    count_per_class = [sum(y == c) for c in clss]
    pred_y = np.argmax(score,axis=1)
    for cls in range(len(clss)):
        indices = np.where(pred_y == cls)[0]
        cls_score = score[indices,cls]
        idx_sort = np.argsort(cls_score) # dec
        add_num = min(int(np.ceil(count_per_class[cls] * add_ratio)),
                      indices.shape[0])
        add_indices[indices[idx_sort[-add_num:]]] = 1
    return add_indices.astype('bool')

def sel_idx_spl(score,untrain_data,add_ratio=0.5):
    y = np.array([label for _,label,_ in untrain_data])
    add_indices = np.zeros(score.shape[0])
    clss = np.unique(y)
    assert score.shape[1] == len(clss)
    count_per_class = [sum(y == c) for c in clss]
    #pred_y = np.argmax(score,axis=1)
    for cls in range(len(clss)):
        indices = np.where(y == cls)[0] #np.where(pred_y == cls)[0]
        cls_score = score[indices,cls]
        idx_sort = np.argsort(cls_score) # dec
        add_num = min(int(np.ceil(count_per_class[cls] * add_ratio)),
                      indices.shape[0])
        #add_num = min(add_num, int(np.ceil(count_per_class[cls] * 0.9))) # [- ~ 0.9N] avoid noisy data
        add_indices[indices[idx_sort[-add_num:]]] = 1
    return add_indices.astype('bool')

def sel_idx_wspl(score,untrain_data,add_ratio=0.5):
    y = np.array([label for _,label,_ in untrain_data])
    add_indices = np.zeros(score.shape[0])
    clss = np.unique(y)
    avgLoss = np.zeros(len(clss))
    num_sort = np.zeros(len(clss))
    num_sort_ = np.zeros(len(clss))
    Hi = np.zeros(len(clss))
    Ci = np.zeros(len(clss))
    curri_num = np.zeros(len(clss))
    curri_lack_num = np.zeros(len(clss))
    assert score.shape[1] == len(clss)
    count_per_class = [sum(y == c) for c in clss]
    #pred_y = np.argmax(score,axis=1)
    for cls in range(len(clss)):
        indices = np.where(y == cls)[0] #np.where(pred_y == cls)[0]
        cls_score = score[indices,cls]
        idx_sort = np.argsort(cls_score) # dec
        add_num = min(int(np.ceil(count_per_class[cls] * add_ratio)),
                      indices.shape[0])
        #add_num = min(add_num, int(np.ceil(count_per_class[cls] * 0.9))) # [- ~ 0.9N] avoid noisy data
        add_indices[indices[idx_sort[-add_num:]]] = 1

        avgLoss[cls] = np.sum(-np.log(cls_score[idx_sort[-add_num:]])) / add_num
        num_sort[cls] = count_per_class[cls]#add_num #np.exp(count_per_class[cls] - max(count_per_class))
        num_sort_[cls] = add_num #np.exp(count_per_class[cls] - max(count_per_class))
        #Hi[cls] = avgLoss[cls] / num_sort[cls]
        Hi[cls] = num_sort[cls] / np.exp(avgLoss[cls]) # 1 / H

    #normalize H to [min, max]
    min_ = min(num_sort)
    max_ = max(num_sort)
    maxHi = max(Hi)
    minHi = min(Hi)
    for cls in range(len(Hi)):
        Hi[cls] = (max_ - min_) * (Hi[cls] - minHi) / (maxHi - minHi) + min_
        Hi[cls] = np.ceil(Hi[cls])

    eng = matlab.engine.start_matlab()
    cmcv = eng.eval('ConsistentCostMatrixGen(%d,10,[%s])' % (len(clss), ','.join(str(x) for x in Hi)), nargout=2)
    cm = np.array(cmcv[0])
    cv = np.array(cmcv[1][0])
    eng.quit()
    #'''
    #cv=cv*(np.sum(Hi)/np.sum(cv*Hi))
    #'''

    sortNum = np.sort(num_sort_)#num_sort_
    indexSortHi = np.argsort(-Hi)
    newCurriculumNum = np.zeros(len(clss))
    for i, n in enumerate(sortNum):
        newCurriculumNum[indexSortHi[i]] = n
        # if newCurriculumNum[indexSortHi[i]] > 60:#40:
        #     newCurriculumNum[indexSortHi[i]] = 60#40

    #avgH = sum(num_sort) / sum(np.exp(avgLoss))
    #newCurriculumNum = [avgH * np.exp(l) for l in avgLoss]

    '''
    min_ = min(num_sort)
    max_ = max(num_sort)
    maxCu = max(newCurriculumNum)
    minCu = min(newCurriculumNum)
    for cls in range(len(newCurriculumNum)):
        newCurriculumNum[cls] = (max_ - min_) * (newCurriculumNum[cls] - minCu) / (maxCu - minCu) + min_
        newCurriculumNum[cls] = np.ceil(newCurriculumNum[cls])
    '''
    subCuccriculumNum = newCurriculumNum - num_sort_
    for i, n in enumerate(subCuccriculumNum):
        if(n > 0):
            subCuccriculumNum[i] = min(n, 100)
        elif(n < 0):
            if(num_sort_[i] < 3 * np.ceil(np.sum(sortNum) / len(sortNum))):
                subCuccriculumNum[i] = 0
            elif(newCurriculumNum[i]  < 3 * np.ceil(np.sum(sortNum) / len(sortNum))):
                subCuccriculumNum[i] = min(0, 3 * np.ceil(np.sum(sortNum) / len(sortNum)) - num_sort_[i])
            #subCuccriculumNum[i] = 0

    cv = cv.astype(float)
    return add_indices.astype('bool'), cv, subCuccriculumNum


def split_dataset(dataset,train_ratio=0.2,seed=0):
    """
    split dataset to train_set and untrain_set
    """
    assert 0 <= train_ratio <= 1
    train_set = []
    untrain_set = []
    np.random.seed(seed)
    pids = np.array([data[1] for data in dataset])
    clss = np.unique(pids)
    #assert len(clss) == 198
    for cls in clss:
        indices = np.where(pids == cls)[0]
        np.random.shuffle(indices)
        train_num = int(np.ceil((len(indices) * train_ratio)))
        train_set += [dataset[i] for i in indices[:train_num]]
        untrain_set += [dataset[i] for i in indices[train_num:]]
    cls1 = np.unique([d[1] for d in train_set])
    cls2 = np.unique([d[1] for d in untrain_set])
    #assert len(cls1) == len(cls2) and len(cls1) == 198
    return train_set,untrain_set
