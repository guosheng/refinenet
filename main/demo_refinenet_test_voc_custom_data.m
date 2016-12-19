
% Author: Guosheng Lin (guosheng.lin@gmail.com)


% perform segmentation prediction on user images.
% specify the location of your images, e.g., setting the following folder:
% ds_config.img_data_dir='../datasets/custom_data';



function demo_refinenet_test_voc_custom_data()


rng('shuffle');

addpath('./my_utils');
dir_matConvNet='../libs/matconvnet_20160516_cuda70_cudnn5_bnchanged/matlab/';
run([dir_matConvNet 'vl_setupnn.m']);


run_config=[];
ds_config=[];

run_config.use_gpu=true;
% run_config.use_gpu=false;
run_config.gpu_idx=1;


model_name=['model_' datestr(now, 'YYYYmmDDHHMMSS') '_evaonly_custom_data'];


ds_name='custom_data';
ds_config.img_data_dir='../datasets/custom_data';


% for voc trained model, control the size of input images
run_config.input_img_short_edge_min=450;
run_config.input_img_short_edge_max=800;


run_config.trained_model_path='../model_trained/voc2012/refinenet_res100';
class_info=gen_class_info_voc();

% another example, using the object parsing model
% run_config.trained_model_path='../model_trained/object_parsing/refinenet_res100';
% class_info=gen_class_info_object_parsing();


% the trained model should match the settings in gen_net_opts_fn and gen_network_fn
run_config.gen_net_opts_fn=@gen_net_opts_model_type1;
run_config.gen_network_fn=@gen_network_main_evaonly_model_type1;


% set the input image scales, useful for multi-scale evaluation
% e.g. using multiple scale settings (1.0 0.8 0.6) and average the resulting score maps.
run_config.input_img_scale=1;


run_config.run_evaonly=true;
ds_config.use_dummy_gt=true;
run_config.use_dummy_gt=ds_config.use_dummy_gt;
ds_config.use_custom_data=true;


ds_config.ds_name=ds_name;
ds_config.class_info=class_info;

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


run_config.learning_rate=0;
run_config.cache_data_mem=false;
run_config.epoch_run_max_task_one_class=0;
run_config.crop_box_size=0;


run_config.net_init_model_path=run_config.trained_model_path;
train_opts=run_config.gen_net_opts_fn(run_config, ds_info.class_info);
net_config=run_config.gen_network_fn(train_opts);

my_net_init_from_existed(run_config, net_config);



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



