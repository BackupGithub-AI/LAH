### Requirements 
Please install the following packages:
- python-3.6.5
- torch-1.2.0+cu92
- numpy-1.14.3
- tqdm-4.26.0
- torchvision-0.4.0+cu92
- torchnet-0.0.4

### Options
- `H`: Hash task
- `p`: if perform the command now
- `datasetname`: the name of dataset
- `image-size`: size of the image
- `epochs`: number of training epochs
- `evaluate`: evaluate model on validation set
- `resume`: path to checkpoint

### VOC 2007
```sh
python perform_task.py -H -p --datasetname voc
```
### COCO
```sh
python perform_task.py -H -p --datasetname coco
```
### MIRFLICKR25k
```sh
python perform_task.py -H -p --datasetname mirflickr25k
```


