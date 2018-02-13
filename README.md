# Multipath RefineNet
This is the source code for our [paper](https://arxiv.org/abs/1611.06612):

```
RefineNet: Multi-Path Refinement Networks for High-Resolution Semantic Segmentation
Guosheng Lin, Anton Milan, Chunhua Shen, Ian Reid
In CVPR 2017
```

A MATLAB based framework for semantic image segmentation and general dense prediction tasks on images.

# Update notes
* 23 Dec 2016:  We did a major update of our code. 
* ***(`new!`)*** **13 Feb 2018**: 
    1.  ***Multi-scale prediction and evaluation code are added.***
    We add demo files for multi-scale prediction, fusion and evaluation. 
    Please refer to the `Testing` section below for more details.
    2.  ***New models available:*** trained models using improved residual pooling.
    Available for these datasets: NYUDv2, Person_Parts, PASCAL_Context, SUNRGBD, ADE20k
    These models will give better performance than the reported results in our CVPR paper.
    3.  ***New models available:*** trained models using ResNet-152 for all 7 datasets.
    Apart from ResNet-101 based models, our ResNet-152 based models of all 7 datasets are now available for download.
    4.  ***Updated trained model for VOC2012:*** this updated model is slightly better than the previous one. 
    We previously uploaded a wrong model.
    5.  All models are now available in Google Drive and Baidu Pan.
    6.  More details are provided on testing, training and implementation. 
    Please refer to `Important notes` in each section below.


# Results
* Results on the CityScapes Dataset (single scale prediction using ResNet-101 based RefineNet)
     [![RefineNet Results on the CityScapes Dataset](http://img.youtube.com/vi/L0V6zmGP_oQ/0.jpg)](https://www.youtube.com/watch?v=L0V6zmGP_oQ)


# Trained models
* ***(`new!`)*** Trained models for the following datasets are available for download.
1. `PASCAL VOC 2012`
2. `Cityscapes`
3. `NYUDv2`
4. `Person_Parts`
5. `PASCAL_Context`
6. `SUNRGBD`
7. `ADE20k`

* Downloads for the above datasets. 
Put the downloaded models in `./model_trained/`    
    * ***(`new!`)*** `RefineNet models using ResNet-101`: [Google Drive](https://drive.google.com/open?id=1U2c1N6QJdzB_8HBgXb7mJ6Qk66JDBHI9) or [Baidu Pan](https://pan.baidu.com/s/1nxf2muP)
    * ***(`new!`)*** `RefineNet models using ResNet-152`: [Google Drive](https://drive.google.com/open?id=1UGhqllXOn_qmDhx_3C9aKCoilZGgycFf) or [Baidu Pan](https://pan.baidu.com/s/1bqDwrWN)
* ***Important notes:***
    *  The trained models of the the following datasets are using improved residual pooling: 
`NYUDv2, Person_Parts, PASCAL_Context, SUNRGBD, ADE20k` 
These models will give better performance than the reported results in our CVPR paper. 
Please also refer to the `Network architecture` section below for more details about improved pooling.
    * The model for `VOC2012` is updated. We previously uploaded a wrong model.


# Network architecture and implementation
* You can find the network graphs that illustrate our architecture in the folder `net_graphs`. Please refer to our paper for more details. 
* We include in this folder the details of improved residual pooling which improves the residual pooling block described in our CVPR paper.
* ***Important notes:***
    * In our up-sampling and fusion layer, we simply use down-sampling for gradient back-propagation. 
    Please refer to the implementation of our fusion layer for details: `My_sum_layer.m`.
    * please refer to our training demo files for more details on implementation


# Installation
* Install [MatConvNet](http://www.vlfeat.org/matconvnet/) and CuDNN. We have modified MatConvNet for our task. A modified copy of MatConvNet is provided in `./lib/`. You need to compile the provided MatConvNet before running. Details of this modification and compiling can be found in `main/my_matconvnet_resnet/README.md`.
* An example script for exporting lib paths is
  `main/my_matlab.sh` 
* Download the following ImageNet pre-trained models and place them in `./model_trained/`:
  `imagenet-resnet-50-dag, imagenet-resnet-101-dag, imagenet-resnet-152-dag` 
They can be downloaded from: [MatConvNet](http://www.vlfeat.org/matconvnet/pretrained/), we also have a copy in [Google Drive](https://drive.google.com/open?id=1Y0s5la0HvEhpNqfKGdJTwNTHzIjKt0VT), [Baidu Pan](https://pan.baidu.com/s/1jJA0kBG).


# Testing

#### 1. Multi-scale prediction and evaluation ***(`new!`)*** 
* First download the trained models and put them in `./model_trained/`. Please refer to the above section `Trained Models`.
* Then refer to the below example scripts for prediction on your images.
    You may need to carefully read through the comments in these demo scripts before using.
    `demo_predict_mscale_[dataset name].m`
* e.g., `demo_predict_mscale_voc.m`, `demo_predict_mscale_nyud`, `demo_predict_mscale_person_parts`
* ***Important notes:***
    * In the default setting, the example scripts will perform multi-scale prediction and fuse multi-scale results to generate final prediction. 
    * The generated masks and scores maps will be saved in your disk.  Note that the score maps are saved in the format of `uint8` with values in [0 255]. You need to cast them into `double` and normalize into [0 1] if you want to use them.
    * The above demo files are able to perform multi-scale prediction and evaluation (e.g., in terms of IoU scores) in a single run.
However, in the default setting, the performance evaluation part is disabled.
Please refer to the comments in the demo files to turn on the performance evaluation. 

#### 2. Single scale prediction and evaluation

*   Single scale prediction and evaluation can be done by changing the scale setting in the multi-scale prediction demo files.
Please refer the the above section for multi-scale prediction.

*  We also provide a simplified version (much less configurations) of prediction demo files. Examples can be found at: `demo_test_simple_voc.m` and `demo_test_simple_city.m`
    These files are only for single scale prediction.



#### 3. Evaluation and fusion on saved results ***(`new!`)*** 
* We provide an example script to perform multi-scale fusion on a number of saved predictions:
    `demo_fuse_saved_prediction_voc.m` : fuse multiple cached predictions to generate the final prediction
* We provide an example script to evaluate the prediction masks saved in your disk:
    `demo_evaluate_saved_prediction_voc.m` : evaluate the segmentation performance, e.g., in terms of IoU scores.



# Training
* The following demo files are provided for training a RefineNet on your own dataset. 
Please carefully read through the comments in the demo files before using this training code. 
    * `demo_refinenet_train.m`
    * `demo_refinenet_train_reduce_learning_rate.m`
* ***Important notes:***
    * We use step-wise policy to reduce learning rate, and more importantly, you need to ***manually*** reduce the learning rate during the training stage. The setting of maximum training iteration just serves as a simple example and it should be adapted to your datasets. More details can be found in the comments of the training demo files.
    * We use the improved version of chained pooling in this training code, which may achieve better result than using the above provided models. 


# Citation
If you find the code useful, please cite our work as

```
@inproceedings{Lin:2017:RefineNet,
  title = {Refine{N}et: {M}ulti-Path Refinement Networks for High-Resolution Semantic Segmentation},
  shorttitle = {RefineNet: Multi-Path Refinement Networks},
  booktitle = {CVPR},
  author = {Lin, G. and Milan, A. and Shen, C. and Reid, I.},
  month = jul,
  year = {2017}
}
```
# License
For academic usage, the code is released under the permissive BSD license. For any commercial purpose, please contact the authors.
