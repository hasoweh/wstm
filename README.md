# WSTM

Official implementation of Weakly Supervised Tree Mapping (WSTM). 

The method uses image level labels to train a pixel-wise classification model using aerial images (RGB + NIR) of tree crowns.

## Installation

Cloning of the repository:

```
git clone https://gitlab.com/wstm.git
cd wstm
```

Installation of the python dependencies via *pip*.
```
pip install -e .
```

## Use

Training and testing scripts can be found in the ```processing_scripts``` folder. 

### Training
Training the model can be done by navigating to the ```wstm/processing_scripts/training``` folder on your local machine and then running the scripts from the command line interface. 

Example:
```
cd wstm/processing_scripts/training

python3 01a_train_image_classifier.py -s 'My_model_weights' -m 'Sem_Deeplab' -v 'cuda:0' -c '/home/user/wstm/processing_scripts/CONFIG/deeplab_image_classifier.json'
```

Scripts are numbered in the order they should be executed. For training script 01, there is an 01a and 01b. 01a trains the image classifier in the typical way using only one loss and the resulting model can be used for generating CAM, SEM or eSEM localization maps. 01b trains the image classifier using two loss values as the second loss is used to train the PCM module. Thus, the resulting model can be used to generate CAM, SEM, eSEM or PCM localization maps. However, if CAM, SEM or eSEM is your target, it is better to use 01a.

Different arguments for script 02 lead to different methods applied for the pseudo-labels.

To get CAM based pseudo-labels:

```
python3 02_get_pseudolabels.py \
-v 'cuda:0' \
-a 8 \
-w '/path/to/model/weights.pt'\
-c '/home/user/wstm/processing_scripts/CONFIG/deeplab_image_classifier.json' \
-b 'train' \
-o '/path/to/output/pseudolabels' \
-m 'Sem_deeplab' \
-t 0.9 \
-r 0.9
```

To get PCM based pseudo-labels:
```
python3 02_get_pseudolabels.py \
-v 'cuda:0' \
-a 8 \
-w '/path/to/model/weights.pt'\
-c '/home/user/wstm/processing_scripts/CONFIG/deeplab_image_classifier.json' \
-b 'train' \
-o '/path/to/output/pseudolabels' \
-m 'Pcm_deeplab' \
-t 0.9 \
-r 0.9 \
-e
```

To get SEM based pseudo-labels:
```
python3 02_get_pseudolabels.py \
-v 'cuda:0' \
-a 8 \
-w '/path/to/model/weights.pt'\
-c '/home/user/wstm/processing_scripts/CONFIG/deeplab_image_classifier.json' \
-b 'train' \
-o '/path/to/output/pseudolabels' \
-m 'Sem_deeplab' \
-t 0.9 \
-r 0.9 \
-s 10
```

To get eSEM based pseudo-labels:
```
python3 02_get_pseudolabels.py \
-v 'cuda:0' \
-a 8 \
-w '/path/to/model/weights.pt'\
-c '/home/user/wstm/processing_scripts/CONFIG/deeplab_image_classifier.json' \
-b 'train' \
-o '/path/to/output/pseudolabels' \
-m 'Sem_deeplab' \
-t 0.9 \
-r 0.9 \
-e \
```

The final segmentation model can be trained as such:
```
python3 03_train_segmentation.py \
-v 'cuda:0' \
-s 'model_weights_segmentation' \
-c '/home/user/wstm/processing_scripts/CONFIG/deeplab_pixel_classifier.json'
```

### Testing
Test scripts can be found in ```wstm/processing_scripts/test```.

Get performance of the image classifier:
```
python3 test_classification.py \
-v 'cuda:0' \
-m 'Sem_deeplab' \
-b 8 \
-c '/home/user/wstm/processing_scripts/CONFIG/deeplab_image_classifier.json' \
-w '/path/to/model/weights.pt'
```

Get performance of the pixel classifier:
```
python3 test_clm_segmentation.py \
-v 'cuda:0' \
-z 8 \
-c '/home/user/wstm/processing_scripts/CONFIG/deeplab_image_classifier.json' \
-w '/path/to/model/weights.pt'
```

Get the segmentation performance of CLMs.

CAM:
```
python3 test_clm_segmentation.py \
-v 'cuda:0' \
-m 'Sem_deeplab' \
-z 8 \
-c '/home/user/wstm/processing_scripts/CONFIG/deeplab_image_classifier.json' \
-w '/path/to/model/weights.pt' \
-r 0.0 \
-p 0.5 
```

PCM:
```
python3 test_clm_segmentation.py \
-v 'cuda:0' \
-m 'Pcm_deeplab' \
-z 8 \
-c '/home/user/wstm/processing_scripts/CONFIG/deeplab_image_classifier.json' \
-w '/path/to/model/weights.pt' \
-r 0.0 \
-p 0.5 \
-e
```

SEM:
```
python3 test_clm_segmentation.py \
-v 'cuda:0' \
-m 'Sem_deeplab' \
-z 8 \
-c '/home/user/wstm/processing_scripts/CONFIG/deeplab_image_classifier.json' \
-w '/path/to/model/weights.pt' \
-r 0.0 \
-p 0.5 \
-s 10
```

eSEM:
```
python3 test_clm_segmentation.py \
-v 'cuda:0' \
-m 'Sem_deeplab' \
-z 8 \
-c '/home/user/wstm/processing_scripts/CONFIG/deeplab_image_classifier.json' \
-w '/path/to/model/weights.pt' \
-r 0.0 \
-p 0.5 \
-e
```


## Using other models included here

### MSGSR-Net
Need to first train the MSGSR model for classification at the image level.

```
cd wstm/processing_scripts/training

python3 01a_train_image_classifier.py -s 'My_model_weights' -m 'MSGSRNET' -v 'cuda:0' -c '/home/user/wstm/processing_scripts/CONFIG/MSGSRNET_image_classifier.json'
```

Second is to obtain the pseudolabels generated by the model.

```
cd wstm/processing_scripts/MSGSR

python3 get_pseudo_labels_msgsr.py -w 'trained_model_weights' -m 'MSGSRNET' -v 'cuda:0' -c '/home/user/wstm/processing_scripts/CONFIG/MSGSRNET_image_classifier.json' -o '/home/user/path/to/save/pseudolabel/files' -b "name of subset (either train or val or test)" -a batch_size
```

Then, train the segmentation model using the pseudolabels
```
cd wstm/processing_scripts

python3 03_train_segmentation.py \
-v 'cuda:0' \
-s 'model_weights_segmentation' \
-c '/home/user/wstm/processing_scripts/CONFIG/deeplab_pixel_classifier.json'
```

### DSRG
First need to train any generic image level classification model.

```
cd wstm/processing_scripts/training

python3 01a_train_image_classifier.py -s 'My_model_weights' -m 'Sem_Deeplab' -v 'cuda:0' -c '/home/user/wstm/processing_scripts/CONFIG/deeplab_image_classifier.json'
```

Then extract the CAMs, but do not process them into pseudolabels.

```
cd wstm/processing_scripts/training/DSRG

python3 save_cams.py -s 'My_model_weights' -m 'Sem_Deeplab' -v 'cuda:0' -c '/home/user/wstm/processing_scripts/CONFIG/deeplab_image_classifier.json' -o '/home/user/path/to/folder/for/cams'
```

Train the segmentation model using DSRG method.

```
cd wstm/processing_scripts/training/DSRG

python3 train_segmentation_dsrg.py -s 'My_model_weights' -m 'Sem_Deeplab' -v 'cuda:0' -c '/home/user/wstm/processing_scripts/CONFIG/deeplab_image_classifier.json' -o '/home/user/path/to/folder/for/cams'
```

