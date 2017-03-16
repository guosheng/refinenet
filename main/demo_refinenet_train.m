
% Author: Guosheng Lin (guosheng.lin@gmail.com)

% This file shows how to perform training on your own dataset.

% 1. Training demo on PASCAL VOC 2012 dataset

% We here demonstrate the training on the original PASCAL VOC 2012 dataset for semantic segmentation.
% Before running this demo file for training, you need to 
% download the PASCAL VOC 2012 dataset and place it at: 
% ../datasets/voc2012_trainval


% 2. Training on other dataset:

% other datasets can be handled in a similar way as the VOC2012 dataset.
% Basically the users need to provide a file to describe your dataset:
% my_gen_ds_info_[your dataset name].m
% please refer to my_gen_ds_info_voc.m for details


% 3. Training instructions:

% Stage-1: using a fixed learning rate for training until the performance
% get stable on the validation set. You need to manually stop the running of this training code.

% Stage-2: chose one cached model, initialize from this model, and using a
% lower learning rate (e.g., multiplied by 0.1) to perform further training. 

% For example, using the following setting for loading an existed model and
% using a lower learning rate:
% run_config.trained_model_path='../cache_data/voc2012_trainval/model_20161219094311_example/model_cache/epoch_70';
% run_config.learning_rate=5e-5;

% Please refer to the following demo file for perform further training with lower learning rate:
% demo_refinenet_train_reduce_learning_rate.m


% 4. Verify your trained model with performance evaluation:

% This demo file shows the training on the original PASCAL VOC 2012 dataset for semantic segmentation.
% This dataset consists of 1464 training images and 1449 validation images. 
% For this dataset, the training will take around 3 days using a Titan-X card, 
% including the training with a decreased learning rate.

% After training, the intersection over union (IoU) score of the trained model is supposed to be around 0.691. 

% If you cannot get similar result, please verify whether you are using our modified version of MatConvNet,
% for which you need to replace the modified files in the original MatConvNet and compile after replacement.
% Details can be found at ./my_matconvnet_resnet/README.txt



function demo_refinenet_train()


rng('shuffle');

addpath('./my_utils');
dir_matConvNet='../libs/matconvnet/matlab';
run(fullfile(dir_matConvNet, 'vl_setupnn.m'));


run_config=[];
ds_config=[];

run_config.use_gpu=true;
% run_config.use_gpu=false;
run_config.gpu_idx=1;

% use a random model name:
model_name=['model_' datestr(now, 'YYYYmmDDHHMMSS')];

% or specify a model_name for training, if the cached snapshot files existed, it will resume training
% model_name='model_20161219094311_example';

% update the model name when reducing learning rate for training, e.g.,
% model_name='model_20161219094311_example_epoch70_low_learn_rate';



ds_name='voc2012_trainval';
gen_ds_info_fn=@my_gen_ds_info_voc;


% control the size of input images
run_config.input_img_short_edge_min=450;
run_config.input_img_short_edge_max=1100;


run_config.input_img_scale=1;


ds_config.use_dummy_gt=false;
run_config.use_dummy_gt=ds_config.use_dummy_gt;
ds_config.use_custom_data=false;


ds_config.ds_name=ds_name;
ds_config.gen_ds_info_fn=gen_ds_info_fn;
ds_config.ds_info_cache_dir=fullfile('../datasets', ds_name);


run_config.root_cache_dir=fullfile('../cache_data', ds_name, model_name);
mkdir_notexist(run_config.root_cache_dir);

run_config.model_name=model_name;

diary_dir=run_config.root_cache_dir;
mkdir_notexist(diary_dir);
diary(fullfile(diary_dir, 'output.txt'));
diary on


run_dir_name=fileparts(mfilename('fullpath'));
[~, run_dir_name]=fileparts(run_dir_name);
run_config.run_dir_name=run_dir_name;
run_config.run_file_name=mfilename();




fprintf('\n\n\n=====================================================================\n');
disp('gen ds_info');

disp('ds_config:');
disp(ds_config);

ds_info=gen_dataset_info(ds_config);

my_diary_flush();


fprintf('\n\n\n=====================================================================\n');
disp('run network');


run_config.run_evaonly=false;


% settings for training:

run_config.trained_model_path=[];
run_config.learning_rate=5e-4;


% init from a cached model
% e.g., can be used for resuming network training with a lower learning rate by
% initilizing from a cached model.
% the following is an example for loading a cached model and using a lower learning
% to continue the training

% please refer to the example script: demo_refinenet_train_reduce_learning_rate.m

% run_config.trained_model_path='../cache_data/voc2012_trainval/model_20161219094311_example/model_cache/epoch_70';
% run_config.learning_rate=5e-5;

run_config.epoch_num=600;

% turn on this option to cache all data into memory, if it's possible
% run_config.cache_data_mem=true;
run_config.cache_data_mem=false;

% random crop training:
run_config.crop_box_size=400;

% for cityscape, using a larger crop:
% run_config.crop_box_size=600;

% evaluate step: do evaluation every 5 epochs, can be set to 10:
run_config.eva_run_step=5;
run_config.snapshot_step=1;

% choose ImageNet pre-trained resnet:
run_config.init_resnet_layer_num=50;
% run_config.init_resnet_layer_num=101;
% run_config.init_resnet_layer_num=152;

% generate network
run_config.gen_network_fn=@gen_network_main;


run_config.gen_net_opts_fn=@gen_net_opts_model_type1;
% run_config.gen_net_opts_fn=@gen_net_opts_model_type2;


% uncomment the following for debug:
% select a subset for both training and evaluation.
% ds_info.train_idxes=ds_info.train_idxes(1:5);
% ds_info.test_idxes=ds_info.train_idxes;
% run_config.snapshot_step=inf;


train_opts=run_config.gen_net_opts_fn(run_config, ds_info.class_info);




disp('train_opts:');
disp(train_opts);

disp('train_opts.eva_param:');
disp(train_opts.eva_param);

my_diary_flush();

imdb=my_gen_imdb(train_opts, ds_info);

data_norm_info=[];
data_norm_info.image_mean=128;

imdb.ref.data_norm_info=data_norm_info;

if run_config.use_gpu
	gpu_num=gpuDeviceCount;
	if gpu_num>=1
		gpuDevice(run_config.gpu_idx);
    else
        error('no gpu found!');
	end
end

[net_config, net_exp_info]=prepare_running_model(train_opts);

% net_config can be changed here before running the network

my_net_tool(train_opts, imdb, net_config, net_exp_info);


fprintf('\n\n--------------------------------------------------\n\n');
disp('results are saved in:');
disp(run_config.root_cache_dir);


my_diary_flush();
diary off


end


