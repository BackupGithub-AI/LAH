#### Requirements

Please install the following packages:
    - python-3.6.5
    - torch-1.2.0+cu92
    - numpy-1.14.3
    - tqdm-4.26.0
    - torchvision-0.4.0+cu92
    - torchnet-0.0.4

#### Options
    - lr: learning rate
    - lrp: factor for learning rate of pretrained layers. The learning rate of the pretrained layers is lr * lrp
    - batch-size: number of images per batch
    - image-size: size of the image
    - epochs: number of training epochs
    - evaluate: evaluate model on validation set
    - resume: path to checkpoint

You can train the model with these commands to different dataset
`python perform_task.py -H -p --datasetname voc`
or
`python perform_task.py -H -p --datasetname coco`
or
`python perform_task.py -H -p --datasetname mirflickr25k`


