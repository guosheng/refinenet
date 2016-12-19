

% Author: Guosheng Lin (guosheng.lin@gmail.com)

% This is an example of resume training with lower learning rate.


function demo_refinenet_train_reduce_learning_rate()


rng('shuffle');

addpath('./my_utils');
dir_matConvNet='../libs/matconvnet_20160516_cuda70_cudnn5_bnchanged/matlab/';
run([dir_matConvNet 'vl_setupnn.m']);


run_config=[];
ds_config=[];

run_config.use_gpu=true;
% run_config.use_gpu=false;
run_config.gpu_idx=1;


% use current time string as model name:
% model_name=['model_' datestr(now, 'YYYYmmDDHHMMSS')];

% specify model_name for training, if the cached files existed, then resume training
% model_name='model_20161219094311_model1';

% update the model name when reducing learning rate for training, e.g.,
% pick the mdoel at epoch 300 for further training.
model_name='model_20161219094311_example_model_epoch300_reduce_learn_rate';


ds_name='voc2012_trainval';
gen_ds_info_fn=@my_gen_ds_info_voc;
run_config.crop_box_size=400;


% ds_name='cityscapes';
% gen_ds_info_fn=@my_gen_ds_info_cityscapes;
% run_config.crop_box_size=600;


run_config.gen_net_opts_fn=@gen_net_opts_model_type1;
run_config.gen_network_fn=@gen_network_main;


run_config.input_img_short_edge_min=450;
run_config.input_img_short_edge_max=1100;
run_config.input_img_scale=1;



run_config.run_evaonly=false;
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



% init from a cached model
% e.g., can be used for resuming network training with a lower learning rate by
% initilizing from a cached model.

% run_config.trained_model_path=[];
% run_config.learning_rate=5e-4;

% here's an example for loading a cached model, e.g., the cached model of epoch 300, 
% and using a lower learning to continue the training.
run_config.trained_model_path='../cache_data/voc2012_trainval/model_20161219094311_example_model/model_cache/epoch_300';
run_config.learning_rate=5e-5;



run_config.cache_data_mem=false;

% turn on this option to cache all data into memory, if it's possible
% run_config.cache_data_mem=true;

run_config.epoch_run_max_task_one_class=100;


run_config.net_init_model_path=run_config.trained_model_path;
train_opts=run_config.gen_net_opts_fn(run_config, ds_info.class_info);
net_config=run_config.gen_network_fn(train_opts);

my_net_init_from_existed(run_config, net_config);





% uncomment the following for debug:
% select a subset for both training and test.
% ds_info.train_idxes=ds_info.train_idxes(1:2);
% ds_info.test_idxes=ds_info.train_idxes;



disp('train_opts:');
disp(train_opts);

disp('train_opts.eva_param:');
disp(train_opts.eva_param);

disp('net_config:');
disp(net_config.ref);

my_diary_flush();


imdb=my_gen_imdb(train_opts, ds_info);

data_norm_info=[];
data_norm_info.use_constant_mean=true;
data_norm_info.constant_mean=128;

imdb.ref.data_norm_info=data_norm_info;


if run_config.use_gpu
	gpu_num=gpuDeviceCount;
	if gpu_num>1
		gpuDevice(run_config.gpu_idx);
    else
        error('no gpu found!');
	end
end

my_net_tool(net_config, imdb, train_opts);


fprintf('\n\n--------------------------------------------------\n\n');
disp('results are saved in:');
disp(run_config.root_cache_dir);


my_diary_flush();
diary off


end





