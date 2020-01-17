
### Requirements
Please, install the following packages
- numpy
- torch-1.2.0+cu92 (newest version at 20191010)
- torchnet
- torchvision-0.4.0+cu92
- tqdm

### Download pretrain models
[Baidu](https://pan.baidu.com/s/17j3lTjMRmXvWHT86zhaaVA)

### Options
- `lr`: learning rate
- `lrp`: factor for learning rate of pretrained layers. The learning rate of the pretrained layers is `lr * lrp`
- `batch-size`: number of images per batch
- `image-size`: size of the image
- `epochs`: number of training epochs
- `evaluate`: evaluate model on validation set
- `resume`: path to checkpoint

### Demo VOC 2007
```sh
python3 demo_voc2007_gcn.py data/voc --image-size 448 --batch-size 32 -e --resume checkpoint/voc/voc_checkpoint.pth.tar
```

### Demo COCO 2014
```sh
python3 demo_coco_gcn.py data/coco --image-size 448 --batch-size 32 -e --resume checkpoint/coco/coco_checkpoint.pth.tar
```

## Citing this repository
If you find this code useful in your research, please consider citing us:

```
=======
# Reading_ML_GCN_sourcecode

ML_GCN.pytorch
PyTorch implementation of Multi-Label Image Recognition with Graph Convolutional Networks, CVPR 2019.

Requirements
Please, install the following packages

numpy
torch-0.3.1
torchnet
torchvision-0.2.0
tqdm
Download pretrain models
checkpoint/coco (GoogleDrive)

checkpoint/voc (GoogleDrive)

or

Baidu

Options
lr: learning rate
lrp: factor for learning rate of pretrained layers. The learning rate of the pretrained layers is lr * lrp
batch-size: number of images per batch
image-size: size of the image
epochs: number of training epochs
evaluate: evaluate model on validation set
resume: path to checkpoint
Demo VOC 2007
python3 demo_voc2007_gcn.py data/voc --image-size 448 --batch-size 32 -e --resume checkpoint/voc/voc_checkpoint.pth.tar
Demo COCO 2014
python3 demo_coco_gcn.py data/coco --image-size 448 --batch-size 32 -e --resume checkpoint/coco/coco_checkpoint.pth.tar
Citing this repository
If you find this code useful in your research, please consider citing us:

>>>>>>> f8b365bfc408cc84742d7ac27185008ce653ac80
@inproceedings{ML_GCN_CVPR_2019,
author = {Zhao-Min, Chen and Xiu-Shen, Wei and Peng, Wang and Yanwen, Guo},
title = {{Multi-Label Image Recognition with Graph Convolutional Networks}},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
year = {2019}
}
<<<<<<< HEAD
```
## Reference
This project is based on https://github.com/durandtibo/wildcat.pytorch

## Tips
=======
Reference
This project is based on https://github.com/durandtibo/wildcat.pytorch

Tips
>>>>>>> f8b365bfc408cc84742d7ac27185008ce653ac80
If you have any questions about our work, please do not hesitate to contact us by emails.
