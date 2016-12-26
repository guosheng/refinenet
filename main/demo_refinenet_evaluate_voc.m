
% Author: Guosheng Lin (guosheng.lin@gmail.com)

% here shows an example of
% evaluating the segmentation performance of a trained model on the voc dataset:

% Before running this demo file for evaluation, you need to 
% download the PASCAL VOC 2012 dataset and place it at: 
% ../datasets/voc2012_trainval


function demo_refinenet_evaluate_voc()


addpath('./my_utils');
dir_matConvNet='../libs/matconvnet/matlab';
run(fullfile(dir_matConvNet, 'vl_setupnn.m'));


run_config=[];
ds_config=[];

run_config.use_gpu=true;
% run_config.use_gpu=false;
run_config.gpu_idx=1;


model_name=['model_' datestr(now, 'YYYYmmDDHHMMSS') '_evaonly'];


ds_name='voc2012_trainval';
gen_ds_info_fn=@my_gen_ds_info_voc;


% for voc trained model, control the size of input images
run_config.input_img_short_edge_min=450;
run_config.input_img_short_edge_max=800;


% set the input image scales, useful for multi-scale evaluation
% e.g. using multiple scale settings (1.0 0.8 0.6) and average the resulting score maps.
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


% do evaluation only
run_config.run_evaonly=true;

% load a trained model:
run_config.trained_model_path='../model_trained/refinenet_res101_voc2012.mat';



% settings for training:

% run_config.learning_rate=5e-4;;

% turn on this option to cache all data into memory, if it's possible
% run_config.cache_data_mem=true;
% run_config.cache_data_mem=false;

% random crop training:
% run_config.crop_box_size=400;

% for cityscape, using a larger crop:
% run_config.crop_box_size=600;

% evaluate step: do evaluation every 10 epochs, can be set to 5:
% run_config.eva_run_step=10;
% run_config.snapshot_step=1;

% choose ImageNet pre-trained resnet:
% run_config.init_resnet_layer_num=50;
% run_config.init_resnet_layer_num=101;
% run_config.init_resnet_layer_num=152;

% generate network
% run_config.gen_network_fn=@gen_network_main;

run_config.gen_net_opts_fn=@gen_net_opts_model_type1;


% uncomment the following for debug:
% select a subset for evaluation.
% ds_info.test_idxes=ds_info.test_idxes(1:10);

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


