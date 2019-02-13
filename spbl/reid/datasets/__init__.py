from __future__ import absolute_import
import warnings

from .cuhk01 import CUHK01
from .cuhk03 import CUHK03
from .dukemtmc import DukeMTMC
from .market1501 import Market1501
from .Market1501_STD import Market1501_STD
from .viper import VIPeR
from .sd_198 import SD_198
from .fsd_198 import FSD_198
from .acne_level4 import ACNE_LEVEL4
from .isic2018_task3 import ISIC2018_Task3
from .sd_260_over100 import SD_260_OVER100
from .sd_260_over200 import SD_260_OVER200
from .indoor import MIT67
from .mnist import MNIST
from .cifar100 import CIFAR100
from .mcr import MCR


__factory = {
    'viper': VIPeR,
    'cuhk01': CUHK01,
    'cuhk03': CUHK03,
    'market1501': Market1501,
    'dukemtmc': DukeMTMC,
    'market1501std': Market1501_STD,
    'sd-198': SD_198,
    'fsd-198': FSD_198,
    'acne_level4': ACNE_LEVEL4,
    'isic2018_task3': ISIC2018_Task3,
    'sd-260-over100': SD_260_OVER100,
    'sd-260-over200': SD_260_OVER200,
    'indoor': MIT67,
    'mnist': MNIST,
    'cifar-100': CIFAR100,
    'mcr': MCR
}


def names():
    return sorted(__factory.keys())


def create(name, root, *args, **kwargs):
    """
    Create a dataset instance.

    Parameters
    ----------
    name : str
        The dataset name. Can be one of 'viper', 'cuhk01', 'cuhk03',
        'market1501', and 'dukemtmc'.
    root : str
        The path to the dataset directory.
    split_id : int, optional
        The index of data split. Default: 0
    num_val : int or float, optional
        When int, it means the number of validation identities. When float,
        it means the proportion of validation to all the trainval. Default: 100
    download : bool, optional
        If True, will download the dataset. Default: False
    """
    if name not in __factory:
        raise KeyError("Unknown dataset:", name)
    return __factory[name](root, *args, **kwargs)


def get_dataset(name, root, *args, **kwargs):
    warnings.warn("get_dataset is deprecated. Use create instead.")
    return create(name, root, *args, **kwargs)
