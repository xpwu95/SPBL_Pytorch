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

iv. Install Python Matlab Engine

Follow the [official instructions](https://www.mathworks.com/help/matlab/matlab-engine-for-python.html) to install python matlab engine. Here is an example using conda:
```shell
cd "matlabroot/extern/engines/python"
python setup.py install
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

The SD-198 and SD-260 datasets can be downloaded from http://xiaopingwu.cn/assets/projects/sd-198/.

It is recommended to symlink the datasets root to `spbl/data/dataset_name/raw/images`.
```
ln -s $YOUR_DATA_ROOT data/dataset_name/raw/images
```
The directories should be arranged like this:
```
SPBL_Pytorch
├──	spbl
|	├── reid
|	├── examples
|	│   ├── data
|	│   │   ├── sd-198
|	│   │   │   ├── raw
|	│   │   │   │   ├── images
|	│   │   │   │   ├── train.txt
|	│   │   │   │   ├── val.txt
|	│   │   ├── sd-260
|	│   │   ├── mit67
|	│   │   ├── caltech101
|	│   │   ├── minist
|	│   │   ├── mlc
```


## Running
```shell
python spbl.py
```

### Configuration
The configuration parameters are mainly in the /examples/cfg.py files. The parameters you most probably change are as follows:

- *input_size*: the size of input image
- *lr*: learning rate
- *batch_size*: batch size
- *workers*: 
- *iter_step*: the pace parameter of the SPBL
- *gamma*: 
- *train_ratio*: the ratio of the initial training set
- *model*: model name
- *dataset*: dataset name
- *class_num*: the class number of the dataset
- *epochs*: total training epochs
- *step_size*: the step size of the learning rate decay
- *add_ratios*: add ratios

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

