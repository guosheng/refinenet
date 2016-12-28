# Multipath RefineNet
This is the source code for our [paper](https://arxiv.org/abs/1611.06612):

RefineNet: Multi-Path Refinement Networks for High-Resolution Semantic Segmentation". Guosheng Lin, Anton Milan, Chunhua Shen, Ian Reid.

A MATLAB based framework for semantic image segmentation and dense prediction.


# Update notes
We did a major update of our code on 23 Dec 2016. If you use an older version, please check out a new copy.


# Installation
* Install [MatConvNet](http://www.vlfeat.org/matconvnet/) and CuDNN. We have modified MatConvNet for our task. A modified copy of MatConvNet is provided in `./lib/`. You need to compile the provided MatConvNet before running. Details of this modification and compiling can be found in `main/my_matconvnet_resnet/README.md`.
* An example script for exporting lib paths is
  * `main/my_matlab.sh` 
* Download the following ImageNet pretrained models from [MatConvNet](http://www.vlfeat.org/matconvnet/pretrained/) and place them in `./model_trained/`.
  * `imagenet-resnet-50-dag, imagenet-resnet-101-dag, imagenet-resnet-152-dag` 

# Network architecture
You can find the network graphs that illustrate our architecture in the folder `net_graphs`. Please refer to our paper for more details.


# Testing

<!-- For now, only the model for the PASCAL VOC (ResNet-101) dataset is provided. More trained models will be available soon.  -->
First download the following trained models and put them in `./model_trained/`, then refer to the example scripts for applying the trained model on test images.

* Trained on PASCAL VOC 2012 dataset for object segmentation (ResNet-101): 
  * [refinenet_res101_voc2012.mat (OneDrive, 426M)](https://1drv.ms/u/s!AmxAc3Al6cbBa42jLruLYscVIiw)
  
* Trained on Person-Part dataset for object parsing (ResNet-101): 
  * [refinenet_res101_person_parts.mat (OneDrive, 426M)](https://1drv.ms/u/s!AmxAc3Al6cbBbZXGYA56ELRjedE)

* Trained on Cityscapes dataset (ResNet-101) for street scene parsing: 
  * [refinenet_res101_cityscapes.mat  (OneDrive, 400M)](https://1drv.ms/u/s!AmxAc3Al6cbBbuYTmQG_dGXAfn4)

* Trained on SUNRGBD dataset (ResNet-101) for indoor scene understanding: 
  * [refinenet_res101_sunrgbd.mat  (OneDrive, 400M)](https://1drv.ms/u/s!AmxAc3Al6cbBcPw22yUv67rEn1Y)

* Trained on PASCAL-Context dataset (ResNet-101) for scene understanding: 
  * [refinenet_res101_pascalcontext.mat  (OneDrive, 426M)](https://1drv.ms/u/s!AmxAc3Al6cbBdiWtNpOaBUM4gu0)

Example scripts for applying these models can be found at: `demo_refinenet_test_example_[dataset name].m`
* e.g., `demo_refinenet_test_example_voc.m`, `demo_refinenet_test_example_person_parts.m`, `demo_refinenet_test_example_cityscapes.m`

We also include a demo script to evaluate the trained models, e.g., in terms of IoU scores:
* `demo_refinenet_evaluate_voc.m` to evaluate the segmentation performance of the trained model;


# Training
The following demo is provided for training a RefineNet on your own dataset
* `demo_refinenet_train.m`

We include the improved version of chained pooling in this code, which may achieve better result than using the above provided models. 


# Citation
If you find the code useful, please cite our work as

```
@article{refinenet,
	title = {Refine{N}et: {M}ulti-Path Refinement Networks for High-Resolution Semantic Segmentation},
	shorttitle = {RefineNet: Multi-Path Refinement Networks},
	url = {https://arxiv.org/abs/1611.06612},
	journal = {arXiv:1611.06612 [cs]},
	author = {Lin, G. and Milan, A. and Shen, C. and Reid, I.},
	month = nov,
	year = {2016},
	note = {arXiv: 1611.06612},
	keywords = {Computer Science - Computer Vision and Pattern Recognition}
}
```
# License
For academic usage, the code is released under the permissive BSD license. For any commercial purpose, please contact the authors.
