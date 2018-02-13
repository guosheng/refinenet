
% Author: Guosheng Lin (guosheng.lin@gmail.com)
% example code for multiscale preidctions, fusion and accuracy evaluation

function demo_predict_mscale_pascalcontext()


rng('shuffle');

addpath('./my_utils');
dir_matConvNet='../libs/matconvnet/matlab';
run(fullfile(dir_matConvNet, 'vl_setupnn.m'));


run_config=[];
ds_config=[];

run_config.use_gpu=true;
% run_config.use_gpu=false;
run_config.gpu_idx=1;


%-------------------------------------------------------------------------------------------------------------------------
% settings for using trained model:

ds_name_subfix='pascalcontext';

% result dir:
result_name=['result_' datestr(now, 'YYYYmmDDHHMMSS') '_predict_custom_data'];
result_dir=fullfile('../cache_data', ['test_examples_' ds_name_subfix], result_name);

% using a trained model:
run_config.trained_model_path='../model_trained/refinenet_res101_pascalcontext.mat'; % resnet101 based refinenet
% run_config.trained_model_path='../model_trained/refinenet_res152_pascalcontext.mat'; % resnet152 based refinenet

% provide class_info for the trained model:
ds_config.class_info=gen_class_info_pascalcontext();

% control the size of input images to avoid excessively small or large images
run_config.input_img_short_edge_min=450;
run_config.input_img_short_edge_max=800;
%-------------------------------------------------------------------------------------------------------------------------



% specify the folder that contains testing images:
img_data_dir='../datasets/example_imgs_pascalcontext';

% providing groundtruth mask folder for evaluation, or set to empty if not avaiable
% if the folder of grountruth masks are provided (an example is shown below), 
% this code will perform evaluation after multiscale prediction and fusion.
gt_mask_dir=[];



%-------------------------------------------------------------------------------------------------------------------------
% settings for accuracy evaluation after multi-scale prediction and fusion:
% the following example specifies images for testing and the groundtruth masks for evaluation:

% Notes: for the evaluation purpose, the image folder should only contain val set images, excluding any training images:
% Notes: the groundtruch masks should follow the class label definition in the varaible specifed above: 'ds_config.class_info'

% uncomment the following lines to enable accuracy evaluation:

% img_data_dir='../datasets/pascalcontext/images_testonly';
% gt_mask_dir='../datasets/pascalcontext/gt_testonly';
%-------------------------------------------------------------------------------------------------------------------------



%-------------------------------------------------------------------------------------------------------------------------
% settings for multi-scale prediction, you can consider 3 scales or 5 scales:

prediction_scales=[0.4 0.6 0.8 1 1.2]; % 5 scales
% prediction_scales=[0.6 0.8 1]; % 3 scales
% prediction_scales=[0.8]; % or only use 1 scale
%-------------------------------------------------------------------------------------------------------------------------




ds_config.img_data_dir=img_data_dir;
result_evaluate_param=[];
result_evaluate_param.gt_mask_dir=gt_mask_dir;
ds_config.result_evaluate_param=result_evaluate_param;

run_config.prediction_scales=prediction_scales;
run_config.model_name=result_name;
run_config.root_cache_dir=result_dir;

do_predict_and_evaluate(run_config, ds_config);

end






function do_predict_and_evaluate(run_config, ds_config)

run_config.gen_net_opts_fn=@gen_net_opts_model_type1;

run_config.run_evaonly=true;
ds_config.use_custom_data=true;
ds_config.use_dummy_gt=true;
run_config.use_dummy_gt=ds_config.use_dummy_gt;

ds_config.ds_name='tmp_data';
mkdir_notexist(run_config.root_cache_dir);


diary_dir=run_config.root_cache_dir;
mkdir_notexist(diary_dir);
diary(fullfile(diary_dir, 'output.txt'));
diary on


run_dir_name=fileparts(mfilename('fullpath'));
[~, run_dir_name]=fileparts(run_dir_name);
run_config.run_dir_name=run_dir_name;
run_config.run_file_name=mfilename();

ds_info=gen_dataset_info(ds_config);
my_diary_flush();

if run_config.use_gpu
	gpu_num=gpuDeviceCount;
	if gpu_num>=1
		gpuDevice(run_config.gpu_idx);
    else
        error('no gpu found!');
	end
end



train_opts=run_config.gen_net_opts_fn(run_config, ds_info.class_info);

imdb=my_gen_imdb(train_opts, ds_info);
data_norm_info=[];
data_norm_info.image_mean=128;
imdb.ref.data_norm_info=data_norm_info;

[net_config, net_exp_info]=prepare_running_model(train_opts);

prediction_scales=run_config.prediction_scales;
scale_num=length(prediction_scales);
predict_result_dirs=cell(scale_num, 1);

for s_idx=1:scale_num

	one_scale=prediction_scales(s_idx);
	one_result_dir=fullfile(run_config.root_cache_dir, sprintf('predict_result_%d', s_idx));
	predict_result_dirs{s_idx}=one_result_dir;

	fprintf('\n\n--------------------------------------------------\n\n');
	fprintf('conduct prediction using image scale: %1.2f  (current scale / total scales: %d/%d) \n', one_scale, s_idx, scale_num);

	train_opts.root_cache_dir=one_result_dir;
	train_opts.input_img_scale=one_scale;
	train_opts.eva_param=update_eva_param_mscale(train_opts.eva_param, train_opts);

	my_net_tool(train_opts, imdb, net_config, net_exp_info);

	fprintf('\n\n--------------------------------------------------\n\n');
	disp('results are saved in:');
	disp(train_opts.root_cache_dir);

	my_diary_flush();

end


if length(predict_result_dirs)>1
    
    fprintf('\n\n--------------------------------------------------\n\n');
    disp('fusing multiscale predictions');

    fuse_param=[];
    fuse_param.predict_result_dirs=predict_result_dirs;
    fuse_param.fuse_result_dir=fullfile(run_config.root_cache_dir, 'predict_result_final_fused');
    fuse_param.cache_fused_score_map=true;
    fuse_multiscale_results(fuse_param, ds_config.class_info);

    fprintf('\n\n--------------------------------------------------\n\n');
    disp('final fused results are saved in:');
    disp(fuse_param.fuse_result_dir);

    my_diary_flush();

    final_prediction_dir=fuse_param.fuse_result_dir;
    
else
    
    final_prediction_dir=predict_result_dirs{1};
end


% evalute the final fused result, if groundtruth is provided
result_evaluate_param=ds_config.result_evaluate_param;
if ~isempty(result_evaluate_param) && ~isempty(result_evaluate_param.gt_mask_dir)
	result_evaluate_param.predict_result_dir=fullfile(final_prediction_dir, 'predict_result_mask');
	fprintf('\n\n--------------------------------------------------\n\n');
	disp('perform evaluation of the predicted masks');	
	result_info=evaluate_predict_results(result_evaluate_param, ds_config.class_info);
    disp_evaluate_result(result_info, ds_config.class_info);
    
    eva_result_cached_file=fullfile(final_prediction_dir, 'eva_result_info.mat');
    fprintf('saving evaluation result to: %s\n', eva_result_cached_file);
    save(eva_result_cached_file, 'result_info');
end


my_diary_flush();
diary off


end


