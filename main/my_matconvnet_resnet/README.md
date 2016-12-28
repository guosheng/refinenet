

We have modified the MatConvNet toolbox to suit our tasks. 

Basically, we change the batch normalization (BN) layers for using specified "moment" parameters for mean and variance, rather than computing from the batch. We need to modify the relevant files in DagNN and the gradient calculation for the implementation of BN layers.

The MatConvNet included in this repository (`./libs/matconvnet`) are already modified and compiled with CUDA 7.0, CuDNN 5.0 and Ubuntu 14.04.
You probably need to re-compile the provided MatConvNet to work with your environment, e.g., using the latest version of CUDA, CuDNN and Ubuntu. 
Please refer to following instruction (in Step 2) for compiling.

If you are using your own MatConvNet downloaded from the official website, you could follow the steps below to update your MatConvNet with our modified files and compile.

Assume that your MatConvNet is placed at: `./libs/matconvnet`

1. Replacing files in the original MatConvNet.

  Modified files are included in this folder. 

  * copy files in the folder `./+dagnn` to `./libs/matconvnet/matlab/+dagnn`

  * copy files in the folder `./@DagNN` to `./libs/matconvnet/matlab/+dagnn/@DagNN`

  * copy files in the folder `./impl` to `./libs/matconvnet/matlab/src/bits/impl`

  We change the gradient calculation in the batch normalization layer for the backward propagation step. Changed lines in these files are marked by "changed for refinenet".
  In our modification, the mean and variance values are considered as provided parameters and irrelevant to the layer input, thus the gradient calculation are modified from the original implementation.


2. After the above file replacement, you need to compile the MatConvNet to apply the changes.

  Please refer to the MatConvNet install instruction page [(link)](http://www.vlfeat.org/matconvnet/install/) for compiling.

  An example compiling MATLAB script is as follows:

  ```
  cd ./libs/matconvnet
  run('./matlab/vl_setupnn.m')

  vl_compilenn('enableGpu', true, ...
               'cudaRoot', '/usr/local/cuda-7.0', ...
               'cudaMethod', 'nvcc', ...
               'enableCudnn', true, ...
               'cudnnRoot', '../../libs/cudnn5')

  %if you don't have a GPU available, run the following to compile for using CPU only:
  %vl_compilenn();
  ```

Contact: Guosheng Lin (guosheng.lin@gmail.com)

