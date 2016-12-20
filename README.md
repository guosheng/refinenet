# Multipath RefineNet
This is the source code for our [paper](https://arxiv.org/abs/1611.06612):

RefineNet: Multi-Path Refinement Networks for High-Resolution Semantic Segmentation". Guosheng Lin, Anton Milan, Chunhua Shen, Ian Reid.

# Installation
* Install [MatConvNet](http://www.vlfeat.org/matconvnet/) and CuDNN.
* A modified copy of MatConvNet is provided in `./lib/`. Details of this modification can be found in `main/my_matconvnet_resnet/README.txt`.
* Example scripts for exporting lib paths are
  * `main/my_matlab.sh` 
* Download the following ImageNet pretrained models from [MatConvNet](http://www.vlfeat.org/matconvnet/pretrained/) and place them in './model_trained'.
  * `imagenet-resnet-50-dag, imagenet-resnet-101-dag, imagenet-resnet-152-dag` 

# Testing
For now, only the model for the PASCAL VOC (ResNet-101) dataset is provided. More pretrained models will be available soon. 
Please see the following demo scripts to run and evaluate it on a test image. Run
* `demo_refinenet_test_voc_custom_data.m` to apply our model to a test image;
* `demo_refinenet_evaluate_voc.m` to evaluate the segmentation performance of the pretrained model;

You can find the network graphs that illustrate our architecture in the folder `net_graphs`.

# Training
The following demo is provided for training a RefineNet on your own dataset
* `demo_refinenet_train.m`


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
