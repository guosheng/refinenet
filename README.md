# Multipath RefineNet
This is the source code for our [paper](https://arxiv.org/abs/1611.06612):

RefineNet: Multi-Path Refinement Networks for High-Resolution Semantic Segmentation". Guosheng Lin, Anton Milan, Chunhua Shen, Ian Reid.

# Installation
* Install [MatConvNet](http://www.vlfeat.org/matconvnet/) and CuDNN.
* A modified copy of MatConvNet is provided in `./lib/`. Details of this modification can be found in `main/my_matconvnet_resnet/README.txt`.
* Example scripts for exporting lib paths are
  * `main/my_matlab.sh` and 
  * `main/my_matlab_disp.sh`

# Testing
For now, only the model for the PASCAL VOC dataset is provided. More pretrained models will be available soon. 
Please see the following demo scripts to run and evaluate it on a test image. Run
* `demo_refinenet_test_voc_custom_data.m` to apply our model to a test image;
* `demo_refinenet_evaluate_voc.m` to evaluate the segmentation performance of the pretrained model;

# Training
The following demo is provided for training a RefineNet on your own dataset
* `demo_refinenet_train.m`


# Citation
If you find the code useful, please cite our work as

```
@article{refinenet,
  author    = {Guosheng Lin and Anton Milan and Chunhua Shen and Ian D. Reid},
  title     = {RefineNet: Multi-Path Refinement Networks for High-Resolution Semantic Segmentation},
  year      = {2016},
  url       = {http://arxiv.org/abs/1611.06612},
} 
```
