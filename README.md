# SPBL_Pytorch
PyTorch implementation of "Self-Paced Balance Learning for Clinical Skin Disease Recognition"

## Installation
This project is based on [mmdetection](https://github.com/open-mmlab/mmdetection). And we add GHM losses in it and modify some code for compatibility.

### Requirements
- Python 3.4+
- PyTorch 0.4.0

### Setup the Environment and Packages
i. Create a new environment
We recommend Anaconda as the package & environment manager. And here is an example:
```shell
conda create -n ghm
conda activate ghm
```

ii. Install PyTorch
Follow the [official instructions](https://pytorch.org/) to install Pytorch. Here is an example using conda:
```shell
conda install pytorch torchvision -c pytorch
```
iii. Install Cython
```shell
conda install cython 
# or "pip install cython"
```

### Install SPBL
Clone the repository
```shell
git clone https://github.com/libuyu/GHM_Detection.git
```

### Prepare Data
It is recommended to symlink the datasets root to `mmdetection/data`.
```
ln -s $YOUR_DATA_ROOT data
```
The directories should be arranged like this:
```
GHM_detection
├──	mmdetection
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
### Script
We provide training and testing scripts and configuration files for both GHM and baseline (focal loss and smooth L1 loss) in the [experiments](https://github.com/libuyu/GHM_Detection/tree/master/experiments) directory. You need specify the path of your own pre-trained model in the config files.

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

## Result

Training using the Res50-FPN backbone and testing on COCO minival.

Method | AP
-- | --
FL + SL1 | 35.6%
GHM-C + SL1 | 35.8%
GHM-C + GHM-R | 37.0%

## License
This project is released under the [MIT license](https://github.com/libuyu/GHM_Detection/blob/master/LICENSE).

## Citation

