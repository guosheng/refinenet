

% This configuration here (gen_net_opts_model_type2.m) applies random sampling for epoch construction and image crop generation.
% This setting doesn't require the classification information, like the class statistics for each image. 

% In contrast, the configuration in file gen_net_opts_model_type1.m uses class value based sampling for epoch construction and image crop generation.


function train_opts=gen_net_opts_model_type2(run_config, class_info)


train_opts=run_config;
train_opts.eva_param=gen_eva_param(run_config, class_info);
train_opts.get_batch_ds_info_fn=@my_get_batch_ds_info;

train_opts.epoch_model_cache_fn=@my_epoch_model_cache;
train_opts.batch_disp_step=0.2;
train_opts.net_run_verbose=false;
train_opts.batch_size=1;
train_opts.bnorm_moment_type_trn='global';
train_opts.bnorm_moment_type_tst='global';


if run_config.run_evaonly
    
    train_opts.do_train=false;
    train_opts.do_eva_val=true;
    
    train_opts.do_cache_model=false;
    train_opts.epoch_num=1;
    
    train_opts.learning_rate=0;
    train_opts.cache_data_mem=false;
    train_opts.eva_epoch_idxes=1;
        
    % turn off the fig plot
    train_opts.fig_plot_step=inf;
    train_opts.refine_config=[];
%     train_opts.net_input_img_scales=[];
        
else
    
    train_opts=do_gen_net_opts_train(train_opts, class_info);
    
end

train_opts.net_run_opts=gen_net_run_opts(train_opts);
my_diary_flush();


end





function eva_param=gen_eva_param(run_config, class_info) 

	eva_param=[];
    
    eva_param.class_info=class_info;
	eva_param.predict_result_dir_mask=fullfile(run_config.root_cache_dir, 'predict_result_mask');
    eva_param.predict_result_dir_full=fullfile(run_config.root_cache_dir, 'predict_result_full');
    eva_param.skip_existing_prediction_eva=false;

    if run_config.run_evaonly
		eva_param.save_predict_mask=true;
	    eva_param.save_predict_result_full=true;
	else
		eva_param.save_predict_mask=false;
	    eva_param.save_predict_result_full=false;
    end
         
    eva_param.eva_densecrf_postprocess=false;

    if eva_param.eva_densecrf_postprocess
        
        error('TODO')
        
        eva_param.predict_result_dir_densecrf=fullfile(train_opts.root_cache_dir, 'predict_result_densecrf');
    end
	

end





function train_opts=do_gen_net_opts_train(train_opts, class_info)



optimizer_param=[];
optimizer_param.weightDecay= 0.0005;
optimizer_param.momentum = 0.9;
optimizer_param.learning_rate=train_opts.learning_rate;
train_opts.optimizer_param=optimizer_param;

train_opts.model_cache_step=train_opts.eva_run_step;
    
train_opts.do_train=true;
train_opts.do_eva_val=true;

train_opts.do_cache_model=true;

if ~isfield(train_opts, 'epoch_num')
    train_opts.epoch_num=700;
end

train_opts.fig_plot_step=1;
train_opts.input_net_do_bp=true;


% setting for control the number of tasks in one epoch
train_opts.epoch_task_subsample=true;
train_opts.epoch_task_num_min=1000;
train_opts.epoch_task_num_max=2000;
train_opts.epoch_run_max_task_one_class=100;

% train_opts.epoch_task_gen_type_train='class_sample';
train_opts.epoch_task_gen_type_train='random';



train_opts.data_aug_config=gen_data_aug_config();
train_opts.data_crop_config=gen_data_crop_config(train_opts);

train_opts=do_config_refinenet(train_opts, class_info);

train_opts.model_cache_dir=fullfile(train_opts.root_cache_dir, 'model_cache');
train_opts.model_snapshot_dir=fullfile(train_opts.model_cache_dir, 'snapshot');


train_opts.eva_run_start=train_opts.eva_run_step;
train_opts.eva_epoch_idxes=gen_sampled_epoch_idxes(...
    train_opts.eva_run_start, train_opts.eva_run_step, train_opts.epoch_num);


if train_opts.do_cache_model

    train_opts.model_cache_epoch_idxes_snapshot=gen_sampled_epoch_idxes(...
        train_opts.snapshot_step, train_opts.snapshot_step, train_opts.epoch_num);

    train_opts.model_cache_epoch_idxes=inf;
    if ~isempty(train_opts.model_cache_step)
        train_opts.model_cache_epoch_idxes=gen_sampled_epoch_idxes(...
            train_opts.model_cache_step, train_opts.model_cache_step, train_opts.epoch_num);
    end

end


end



function net_run_opts=gen_net_run_opts(train_opts)


net_run_opts=gen_net_run_opts_basic();

net_run_opts.global_init_fn=@cnn_global_init_fn;
net_run_opts.batch_info_disp_fn=@cnn_batch_info_disp;
net_run_opts.epoch_evaluate_fn=@cnn_epoch_evaluate;
net_run_opts.epoch_run_config_fn=@my_epoch_run_config;
net_run_opts.batch_run_config_fn=@my_batch_run_config;
net_run_opts.net_progress_disp_fn=@my_net_progress_disp;


end






function sampled_epoch_idxes=gen_sampled_epoch_idxes(epoch_start, sample_step, epoch_num)

    sampled_epoch_idxes=sample_step:sample_step:epoch_num;
    sampled_epoch_idxes=sampled_epoch_idxes(sampled_epoch_idxes>epoch_start);
        
    sampled_epoch_idxes=[epoch_start sampled_epoch_idxes];
    if ~any(sampled_epoch_idxes==epoch_num)
        sampled_epoch_idxes=[sampled_epoch_idxes epoch_num];
    end

end






function data_aug_config=gen_data_aug_config()


data_aug_config=[];

data_aug_config.do_aug=true;

% random scale
data_aug_config.aug_scales=[0.7:0.1:1.3];

% random horizontal flip
data_aug_config.aug_flips=[false true];


end


function data_crop_config=gen_data_crop_config(run_config)

data_crop_config=[];

data_crop_config.do_crop=true;

% data_crop_config.gen_crop_point_type='class_sample';
data_crop_config.gen_crop_point_type='random';

data_crop_config.crop_box_step_ratio=0.2;
data_crop_config.crop_box_size=run_config.crop_box_size;



end




function featnet_config=gen_featnet_config_imgraw()
    
    featnet_config=[];
    featnet_config.featnet_type='imgraw';
    featnet_config.conv_block_num=1;
    featnet_config.conv_num_one_block=2;
    featnet_config.filter_num_blocks=[64 64];
    featnet_config.output_dim=featnet_config.filter_num_blocks(end);
    featnet_config.input_img_scale=1;
    
    one_refine_config=[];
    one_refine_config.input_adapt_dim=featnet_config.filter_num_blocks(end);
    one_refine_config.input_name=sprintf('path_raw_out%d', 1);
    featnet_config.refine_config_paths{1}=one_refine_config;
    
    featnet_config.lr_multiplier=1;
            
end



function featnet_config=gen_featnet_config_resnet_custom(train_opts, init_resnet_layer, img_scale, output_path_num, start_stage_idx)

    
    featnet_config=[];
    featnet_config.featnet_type='resnet';
    featnet_config.lr_multiplier=train_opts.input_featnet_lr_multiplier;

    input_adapt_dims=[512 256 256 256];

    featnet_config.input_img_scale=img_scale;
    featnet_config.init_output_path_idx=start_stage_idx;
    
    
    refine_config_paths=cell(output_path_num, 1);
    for o_idx=1:output_path_num

    	out_path_idx=start_stage_idx+o_idx-1;

    	path_str=sprintf('p_ims%.1f_outl%d', img_scale, out_path_idx);
		path_str(strfind(path_str, '.'))='d';

        one_refine_config=[];
        one_refine_config.input_adapt_dim=input_adapt_dims(o_idx);
        one_refine_config.input_name=path_str;
        refine_config_paths{o_idx}=one_refine_config;
    end
    
    
    featnet_config.output_path_num=output_path_num;
    featnet_config.refine_config_paths=refine_config_paths;
    
    featnet_config.init_resnet_layer=init_resnet_layer;
    featnet_config.pre_trained_model_file=sprintf('../model_trained/imagenet-resnet-%d-dag.mat', init_resnet_layer);
        
end







function refine_config=gen_refine_config_basic()


	refine_config=[];
            
    refine_config.use_chained_pool=true;
    
    % set the pooling number to 2 or 4:
    % refine_config.chained_pool_num=2;
    refine_config.chained_pool_num=4;
    
    refine_config.chained_pool_size=5;
    
    if ~refine_config.use_chained_pool
        refine_config.chained_pool_num=0;
    end
        
    refine_config.refine_block_conv_num_mainflow=3;
    refine_config.adapt_conv_num=2;

end





function train_opts=do_config_refinenet(train_opts, class_info)


train_opts.input_featnet_lr_multiplier=0.1;
resnet_layer_num=train_opts.init_resnet_layer_num;
  
input_featnet_configs=cell(0);

% use ResNet50/ResNet101/ResNet152
input_featnet_configs{1}=gen_featnet_config_resnet_custom(train_opts, resnet_layer_num, 1.2, 4, 1);

% 2-scale setting:
% input_featnet_configs{1}=gen_featnet_config_resnet_custom(train_opts, 50, 0.6, 4, 1);
% input_featnet_configs{2}=gen_featnet_config_resnet_custom(train_opts, 50, 1.2, 4, 1);


% using an extra feature net connected directly from input image
% input_featnet_configs{end+1, 1}=gen_featnet_config_imgraw();


refine_config=gen_refine_config_basic();

refine_config_paths=cell(0);
featnet_num=length(input_featnet_configs);
net_input_img_scales=zeros(featnet_num, 1);
for featnet_idx=1:featnet_num
    
    one_featnet_config=input_featnet_configs{featnet_idx};
    one_featnet_config.featnet_idx=featnet_idx;
    input_featnet_configs{featnet_idx}=one_featnet_config;
    
    net_input_img_scales(featnet_idx)=one_featnet_config.input_img_scale;
    refine_config_paths=cat(1, refine_config_paths, one_featnet_config.refine_config_paths);
end
train_opts.net_input_img_scales=net_input_img_scales;
refine_config.refine_config_paths=refine_config_paths;


refine_config.path_group_ids=[1 2 3 4];
    
% 2-scale setting:
%refine_config.path_group_ids=[1 2 3 4 1 2 3 4];

assert(length(refine_config_paths)==length(refine_config.path_group_ids));

loss_config.lr_multiplier=1;
loss_config.lossgroup_conv_num=0;
loss_config.class_num=class_info.class_num;

train_opts.refine_config=refine_config;
train_opts.input_featnet_configs=input_featnet_configs;
train_opts.loss_config=loss_config;


end



