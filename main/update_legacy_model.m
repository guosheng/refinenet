

% update the model which are trained with the legacy code to an updated format, 
% which generates a new model which is compliant with the current code.

% One change in the current code: 
% in the legacy code, a model is saved into multiple mat files, while the
% current code save the model into one mat file.


function update_legacy_model()


rng('shuffle');

addpath('./my_utils');
dir_matConvNet='../libs/matconvnet/matlab/';
run([dir_matConvNet 'vl_setupnn.m']);


fprintf('\n\n\n=====================================================================\n');
disp('update legacy model...');


run_config=[];
run_config.root_cache_dir='../cache_data/tmp';

run_config.input_img_short_edge_min=450;
run_config.input_img_short_edge_max=800;


% class_info=gen_class_info_voc();
% run_config.trained_model_path='../model_trained/voc2012_refinenet_res101_legacy';

% class_info=gen_class_info_person_parts();
% run_config.trained_model_path='../model_trained/person_parts_refinenet_res101_legacy';

% class_info=gen_class_info_cityscapes();
% run_config.trained_model_path='../model_trained/cityscapes_refinenet_res101_legacy';

% class_info=gen_class_info_sunrgbd();
% run_config.trained_model_path='../model_trained/sunrgbd_refinenet_res101_legacy';

class_info=gen_class_info_pascalcontext();
run_config.trained_model_path='../model_trained/pascalcontext_refinenet_res101_legacy';

run_config.trained_model_path_updated=[run_config.trained_model_path '_updated'];


run_config.run_evaonly=false;

% settings for training:
run_config.learning_rate=0;
run_config.cache_data_mem=false;
run_config.epoch_run_max_task_one_class=0;
run_config.crop_box_size=0;

run_config.eva_run_step=10;
run_config.model_cache_step=10;
run_config.snapshot_step=1;

% run_config.init_resnet_layer_num=50;
run_config.init_resnet_layer_num=101;
% run_config.init_resnet_layer_num=152;

run_config.gen_net_opts_fn=@gen_net_opts_model_type1;
run_config.gen_network_fn=@gen_network_main;

train_opts=run_config.gen_net_opts_fn(run_config, class_info);

my_diary_flush();

net_config=prepare_running_model_legacy(train_opts);

mkdir_notexist(run_config.trained_model_path_updated);
net_config.ref.cache_filename='net-config.mat';
cnn_save_model(run_config.trained_model_path_updated, net_config, []);


fprintf('\n\n--------------------------------------------------\n\n');
disp('updated model is saved in:');
disp(run_config.trained_model_path_updated);


end



function [net_config, net_exp_info]=prepare_running_model_legacy(train_opts)


fprintf('\n\n-------------------------------------------------------------\n\n');
disp('prepare_running_model...');

net_exp_info=[];

net_config=train_opts.gen_network_fn(train_opts);

fprintf('load legacy model:\n');
disp(train_opts.trained_model_path);
my_load_legacy_model(train_opts.trained_model_path, net_config);
          

fprintf('\n\n-------------------------------------------------------------\n\n');

end



