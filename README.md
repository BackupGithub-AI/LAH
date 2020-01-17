### Requirements 
Please install the following packages:
- python-3.6.5
- torch-1.2.0+cu92
- numpy-1.14.3
- tqdm-4.26.0
- torchvision-0.4.0+cu92
- torchnet-0.0.4

### Options
- `lr`: learning rate
- `lrp`: factor for learning rate of pretrained layers. The learning rate of the pretrained layers is `lr * lrp`
- `batch-size`: number of images per batch
- `image-size`: size of the image
- `epochs`: number of training epochs
- `evaluate`: evaluate model on validation set
- `resume`: path to checkpoint

### VOC 2007
```sh
python demo_voc2007_gcn.py data/voc2007/ --image-size 448 --batch-size 32
--HASH_TASK --IS_USE_MFB --NORMED --IS_USE_IOU -t -v --linear_intermediate 358 --pooling_stride 2 --epochs 15
--threshold_p 0.15 --HASH_BIT 48
```
### COCO
```sh
python demo_coco_gcn.py data/coco/ --image-size 448 --batch-size 32
--HASH_TASK --IS_USE_MFB --NORMED --IS_USE_IOU -t -v --linear_intermediate 358 --pooling_stride 2 --epochs 15
--threshold_p 0.15 --HASH_BIT 48
```
### MIRFLICKR25k
```sh
python demo_mirflickr25k_gcn.py data/mirflickr25k/ --image-size 448 --batch-size 32
--HASH_TASK --IS_USE_MFB --NORMED --IS_USE_IOU -t -v --linear_intermediate 358 --pooling_stride 2 --epochs 15
--threshold_p 0.15 --HASH_BIT 48
```


