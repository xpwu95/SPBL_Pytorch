from __future__ import print_function, absolute_import
import os.path as osp

from ..utils.data import Dataset
from ..utils.osutils import mkdir_if_missing
from ..utils.serialization import write_json


class MIT67(Dataset):
    #url = 'cv.nankai.edu.cn/projects/sd-198/SD-198.zip'
    #md5 = ''

    def __init__(self, root, split_id=0, num_val=100, download=True):
        super(MIT67, self).__init__(root, split_id=split_id)

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

        # Download the raw zip file
        '''
        fpath = osp.join(raw_dir, 'Market-1501-v15.09.15.zip')
        if osp.isfile(fpath) and \
                hashlib.md5(open(fpath, 'rb').read()).hexdigest() == self.md5:
            print("Using downloaded file: " + fpath)
        else:
            raise RuntimeError("Please download the dataset manually from {} "
                               "to {}".format(self.url, fpath))
        # Extract the file
        exdir = osp.join(raw_dir, 'Market-1501-v15.09.15')
        if not osp.isdir(exdir):
            print("Extracting zip file")
            with ZipFile(fpath) as z:
                z.extractall(path=raw_dir)
        '''
        exdir = osp.join(raw_dir, 'images')
        mkdir_if_missing(exdir)

        # Format
        images_dir = osp.join(self.root, 'images')
        mkdir_if_missing(images_dir)

        # 1501 identities (+1 for background) with 6 camera views each
        #  _ _len_ = 1502
        # [[],[],...,[]] , []
        #identities = [[[] for _ in range(6)] for _ in range(1502)]
        identities = [[[] for _ in range(1)] for _ in range(67)]
        identities_train = [[[] for _ in range(1)] for _ in range(67)]
        label_train = [[[] for _ in range(1)] for _ in range(67)]
        identities_test = [[[] for _ in range(1)] for _ in range(67)]
        label_test = [[[] for _ in range(1)] for _ in range(67)]

        def register_w(subdir):
            pids = set()
            file = open(osp.join('/home/ubuntu1/wxp/TNNLS2018/open-reid-master_sd198/examples/data/indoor/raw', subdir), 'r')
            while 1:
                line = file.readline()
                if not line:
                    break
                # print(line)
                imgdir, pid = line.split()
                pid = int(pid)
                cam = 0
                pids.add(pid)
                fname = ('{:08d}_{:02d}_{:04d}.jpg'
                         .format(pid, cam, len(identities[pid][cam])))
                identities[pid][cam].append(fname)
                if(subdir == 'train_7.txt'):
                    identities_train[pid][cam].append(fname)
                    label_train[pid][cam].append(pid)
                if(subdir == 'val_7.txt'):
                    identities_test[pid][cam].append(fname)
                    label_test[pid][cam].append(pid)
                fpath = osp.join(raw_dir, 'images', imgdir)
                shutil.copy(fpath, osp.join(images_dir, fname))
            file.close()
            return pids

        '''
        def register(subdir, pattern=re.compile(r'([-\d]+)_c(\d)')):
            fpaths = sorted(glob(osp.join(exdir, subdir, '*.jpg')))
            pids = set()
            for fpath in fpaths:
                fname = osp.basename(fpath)
                pid, cam = map(int, pattern.search(fname).groups())
                if pid == -1: continue  # junk images are just ignored
                assert 0 <= pid <= 198  # pid == 0 means background
                assert 1 <= cam <= 1
                cam -= 1
                pids.add(pid)
                fname = ('{:08d}_{:02d}_{:04d}.jpg'
                         .format(pid, cam, len(identities[pid][cam])))
                identities[pid][cam].append(fname)
                shutil.copy0_256x128_sd198(fpath, osp.join(images_dir, fname))
            return pids
        '''
        #trainval_pids = register('bounding_box_train')
        trainval_pids = register_w('train_7.txt')
        #gallery_pids = register('bounding_box_test')
        gallery_pids = register_w('val_7.txt')
        #query_pids = register('query')
        query_pids = []#register_w('query_none')
        #assert query_pids <= gallery_pids
        #assert trainval_pids.isdisjoint(gallery_pids)

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
