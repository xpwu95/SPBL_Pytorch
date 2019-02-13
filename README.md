# SPBL_Pytorch
PyTorch implementation of "Self-Paced Balance Learning for Clinical Skin Disease Recognition"

## Installation
This project is based on [spaco](https://github.com/Flowerfan/open-reid) and [Open-Reid](https://github.com/Cysu/open-reid.git). And we add imbalanced learning methods in it and modify some code for compatibility.

### Requirements
- Python 3.4+
- PyTorch 0.4.0

### Setup the Environment and Packages
i. Create a new environment
We recommend Anaconda as the package & environment manager. And here is an example:
```shell
conda create -n spbl
conda activate spbl
```

ii. Install PyTorch
Follow the [official instructions](https://pytorch.org/) to install Pytorch. Here is an example using conda:
```shell
conda install pytorch=0.4.0 torchvision -c pytorch
```
iii. Install Cython
```shell
conda install cython 
# or "pip install cython"
```

### Install SPBL
i.Clone the repository
```shell
git clone https://github.com/xpwu95/SPBL_Pytorch.git
```
ii. Compile extensions
```shell
cd open-reid
python setup.py install
```

### Prepare Data
It is recommended to symlink the datasets root to `spbl/data`.
```
ln -s $YOUR_DATA_ROOT data
```
The directories should be arranged like this:
```
SPBL_Pytorch
├──	spbl
|	├── mmdet
|	├── tools
|	├── configs
|	├── data
|	│   ├── coco
|	│   │   ├── annotations
|	│   │   ├── train2017
|	│   │   ├── val2017
|	│   │   ├── test2017
|	│   ├── VOCdevkit
|	│   │   ├── VOC2007
|	│   │   ├── VOC2012
```


## Running
```shell
bash spbl.sh
```

### Configuration
The configuration parameters are mainly in the cfg_*.py files. The parameters you most probably change are as follows:

- *work_dir*: the directory for current experiment
- *datatype*: data set name (coco, voc, etc.)
- *data_root*: Root for the data set
- *model.pretrained*: the path to the ImageNet pretrained backbone model
- *resume_from*: path or checkpoint file if resume
- *train_cfg.ghmc*: params for GHM-C loss
	- *bins*: unit region numbers
	- *momentum*: moving average parameter \alpha
- *train_cfg.ghmr*: params for GHM-R loss
	- *mu*: the \mu for ASL1 loss
	- *bins*, *momentum*: similar to ghmc 
- *total_epochs*, *lr_config.step*: set the learning rate decay strategy

### Loss Functions
* The GHM-C and GHM-R loss functions are available in [ghm_loss.py](https://github.com/libuyu/GHM_Detection/blob/master/mmdetection/mmdet/core/loss/ghm_loss.py).
* The code works for pytorch 0.4.1 and later version.

## Results

Dataset | Precision | Recall | F1 | G-mean | MAUC | Accuracy
-- | -- | -- | -- | -- | -- | --
SD-198 | 71.4±1.7 | 65.7±1.6 | 66.2±1.6 | 42.8±4.0 | 68.5±1.6 | 67.8±1.8
SD-260 | 59.9±1.6 | 48.2±1.1 | 51.0±0.9 | 19.6±1.1 | 64.8±1.2 | 65.1±0.8

Dataset | MIT-67 | Caltech-101 | MINIST | MLC 
-- | -- | -- | -- | --
Accuracy | 64.1±0.5 | 88.6±0.4 | 99.0±0.1 | 72.0

## License
This project is released under the [MIT license](https://github.com/libuyu/GHM_Detection/blob/master/LICENSE).

## Citation

