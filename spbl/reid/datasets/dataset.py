from __future__ import print_function, absolute_import
import os.path as osp

from ..utils.data import Dataset
from ..utils.osutils import mkdir_if_missing
from ..utils.serialization import write_json
from examples.cfg import config
import os
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = curPath[:curPath.find("spbl")+len("spbl")]

class Dataset(Dataset):

    def __init__(self, root, split_id=0, num_val=100, download=True):
        super(Dataset, self).__init__(root, split_id=split_id)

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted. " +
                               "You can use download=True to download it.")

        self.load(num_val)

    def download(self):
        if self._check_integrity():
            print("Files already downloaded and verified")
            return

        import re
        import hashlib
        import shutil
        from glob import glob
        from zipfile import ZipFile

        raw_dir = osp.join(self.root, 'raw')
        mkdir_if_missing(raw_dir)

        exdir = osp.join(raw_dir, 'images')
        mkdir_if_missing(exdir)

        # Format
        images_dir = osp.join(self.root, 'images')
        mkdir_if_missing(images_dir)

        # 1501 identities (+1 for background) with 6 camera views each
        #  _ _len_ = 1502
        # [[],[],...,[]] , []
        #identities = [[[] for _ in range(6)] for _ in range(1502)]
        identities = [[[] for _ in range(1)] for _ in range(config.class_num)]
        identities_train = [[[] for _ in range(1)] for _ in range(config.class_num)]
        label_train = [[[] for _ in range(1)] for _ in range(config.class_num)]
        identities_test = [[[] for _ in range(1)] for _ in range(config.class_num)]
        label_test = [[[] for _ in range(1)] for _ in range(config.class_num)]

        def register_w(subdir):
            pids = set()
            file = open(osp.join(rootPath, '/examples/data/', config.dataset, '/raw', subdir), 'r')
            while 1:
                line = file.readline()
                if not line:
                    break
                imgdir, pid = line.split()
                pid = int(pid)
                cam = 0
                pids.add(pid)
                fname = ('{:08d}_{:02d}_{:04d}.jpg'
                         .format(pid, cam, len(identities[pid][cam])))
                identities[pid][cam].append(fname)
                if(subdir == 'train.txt'):
                    identities_train[pid][cam].append(fname)
                    label_train[pid][cam].append(pid)
                if(subdir == 'val.txt'):
                    identities_test[pid][cam].append(fname)
                    label_test[pid][cam].append(pid)
                fpath = osp.join(raw_dir, 'images', imgdir)
                shutil.copy(fpath, osp.join(images_dir, fname))
            file.close()
            return pids

        trainval_pids = register_w('train.txt')
        gallery_pids = register_w('val.txt')
        query_pids = []

        # Save meta information into a json file
        meta = {'name': 'sd_198', 'shot': 'multiple', 'num_cameras': 6,
                'identities': identities,
                'identities_train': identities_train, 'identities_test': identities_test,
                'label_train': label_train, 'label_test': label_test}
        write_json(meta, osp.join(self.root, 'meta.json'))

        # Save the only training / test split
        splits = [{
            'trainval': sorted(list(trainval_pids)),
            'query': sorted(list(query_pids)),
            'gallery': sorted(list(gallery_pids))}]
        write_json(splits, osp.join(self.root, 'splits.json'))
