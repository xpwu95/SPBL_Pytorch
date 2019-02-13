from __future__ import print_function
import os.path as osp

import numpy as np

from ..serialization import read_json


def _pluck(identities, indices, relabel=False):
    ret = []
    for index, pid in enumerate(indices):
        pid_images = identities[pid]
        for camid, cam_images in enumerate(pid_images):
            for fname in cam_images:
                name = osp.splitext(fname)[0]
                x, y, _ = map(int, name.split('_'))
                assert pid == x and camid == y
                if relabel:
                    ret.append((fname, index, camid))
                else:
                    ret.append((fname, pid, camid))
    return ret


class Dataset_ISIC(object):
    def __init__(self, root, split_id=0):
        self.root = root
        self.split_id = split_id
        self.meta = None
        self.split = None
        self.train, self.val, self.trainval = [], [], []
        self.query, self.gallery = [], []
        self.num_train_ids, self.num_val_ids, self.num_trainval_ids = 0, 0, 0

    @property
    def images_dir(self):
        return osp.join(self.root, 'images')

    def load(self, num_val=0.3, verbose=True):
        splits = read_json(osp.join(self.root, 'splits.json')) # class id cotained in query, trainval and gallery subset
        if self.split_id >= len(splits):
            raise ValueError("split_id exceeds total splits {}"
                             .format(len(splits)))
        self.split = splits[self.split_id] # type transform of per subsets

        # Randomly split train / val
        trainval_pids = np.asarray(self.split['trainval'])
        #np.random.shuffle(trainval_pids)
        num = len(trainval_pids)
        num_val = 0.3 # num_val = 100
        if isinstance(num_val, float):
            num_val = int(round(num * num_val))
        if num_val >= num or num_val < 0:
            raise ValueError("num_val exceeds total identities {}"
                             .format(num))
        train_pids = sorted(trainval_pids)#[:-num_val])
        val_pids = sorted(trainval_pids)#[-num_val:])

        self.meta = read_json(osp.join(self.root, 'meta.json'))
        identities       = self.meta['identities']
        identities_train = self.meta['identities_train']
        identities_test  = self.meta['identities_test']

        identities_final_val = self.meta['identities_final_val']
        identities_final_test  = self.meta['identities_final_test']

        #self.train = _pluck(identities, train_pids, relabel=True)
        self.train = _pluck(identities_train, train_pids, relabel=True)  #what we need to train net
        #self.val = _pluck(identities, val_pids, relabel=True)
        self.val = _pluck(identities_test, val_pids, relabel=True)       #what we need to test net

        self.final_val = _pluck(identities_final_val, val_pids, relabel=True)       #what we need to final val net
        self.final_test = _pluck(identities_final_test, val_pids, relabel=True)       #what we need to final test net

        self.trainval = _pluck(identities, trainval_pids, relabel=True)
        self.query = _pluck(identities, self.split['query'])
        self.gallery = _pluck(identities, self.split['gallery'])
        self.num_train_ids = len(train_pids)
        self.num_val_ids = len(val_pids)
        self.num_trainval_ids = len(trainval_pids)

        if verbose:
            print(self.__class__.__name__, "dataset loaded")
            print("  subset   | # ids | # images")
            print("  ---------------------------")
            print("  train    | {:5d} | {:8d}"
                  .format(self.num_train_ids, len(self.train)))
            print("  val      | {:5d} | {:8d}"
                  .format(self.num_val_ids, len(self.val)))

            print("  final_val| {:5d} | {:8d}"
                  .format(self.num_val_ids, len(self.final_val)))
            print(" final_test| {:5d} | {:8d}"
                  .format(self.num_val_ids, len(self.final_test)))

            print("  trainval | {:5d} | {:8d}"
                  .format(self.num_trainval_ids, len(self.trainval)))
            print("  query    | {:5d} | {:8d}"
                  .format(len(self.split['query']), len(self.query)))
            print("  gallery  | {:5d} | {:8d}"
                  .format(len(self.split['gallery']), len(self.gallery)))

    def _check_integrity(self):
        return osp.isdir(osp.join(self.root, 'images')) and \
               osp.isfile(osp.join(self.root, 'meta.json')) and \
               osp.isfile(osp.join(self.root, 'splits.json'))
