


function train_opts=gen_net_opts_model_type1(run_config, class_info)



train_opts=run_config;


train_opts.eva_param=gen_eva_param(run_config, class_info);


if run_config.run_evaonly
    
    train_opts.do_train=false;
    train_opts.do_cache_model=false;
    train_opts.epoch_num=1;
    
    train_opts.eva_run_step=1;
    train_opts.eva_run_start=1;
    train_opts.fig_plot_step=inf;
    
else
    
    % for training:
    
    train_opts.do_train=true;
    train_opts.do_cache_model=true;
    train_opts.epoch_num=2000;
    
    train_opts.eva_run_step=10;
    train_opts.eva_run_start=10;
    train_opts.fig_plot_step=1;
    
    train_opts.model_cache_step=10;
    train_opts.snapshot_step=1;

    %debug:
%     train_opts.model_cache_step=20;
%     train_opts.snapshot_step=inf;
    
end



train_opts.do_eva_val=true;
train_opts.batch_disp_step=0.2;
train_opts.batch_size=1;
train_opts.input_net_do_bp=true;
train_opts.net_run_verbose=false;




train_opts.epoch_task_subsample=true;
train_opts.epoch_task_num_min=1000;
train_opts.epoch_task_num_max=2000;

train_opts.epoch_task_gen_type_train='class_sample';
% train_opts.epoch_task_gne_type_train='random';


train_opts.get_batch_ds_info_fn=@my_get_batch_ds_info;

train_opts.data_aug_config=gen_data_aug_config();
train_opts.data_crop_config=gen_data_crop_config(run_config);

train_opts=do_config_refinenet(train_opts, class_info);

train_opts.model_cache_dir=fullfile(train_opts.root_cache_dir, 'model_cache');
train_opts.model_snapshot_dir=fullfile(train_opts.model_cache_dir, 'snapshot');



train_opts.eva_epoch_idxes_val=gen_eva_run_epoch_idxes(...
    train_opts.eva_run_start, train_opts.eva_run_step, train_opts.epoch_num);
train_opts.eva_epoch_idxes_trn=train_opts.eva_epoch_idxes_val;


if train_opts.do_cache_model

    train_opts.model_cache_epoch_idxes_snapshot=gen_eva_run_epoch_idxes(...
        train_opts.snapshot_step, train_opts.snapshot_step, train_opts.epoch_num);

    train_opts.model_cache_epoch_idxes=inf;
    if ~isempty(train_opts.model_cache_step)
        train_opts.model_cache_epoch_idxes=gen_eva_run_epoch_idxes(...
            train_opts.model_cache_step, train_opts.model_cache_step, train_opts.epoch_num);
    end

end


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






function eva_run_epoch_idxes=gen_eva_run_epoch_idxes(eva_run_start, eva_run_step, epoch_num)

    eva_run_epoch_idxes=eva_run_step:eva_run_step:epoch_num;
    eva_run_epoch_idxes=eva_run_epoch_idxes(eva_run_epoch_idxes>eva_run_start);
        
    eva_run_epoch_idxes=[eva_run_start eva_run_epoch_idxes];
    if ~any(eva_run_epoch_idxes==epoch_num)
        eva_run_epoch_idxes=[eva_run_epoch_idxes epoch_num];
    end

end






function data_aug_config=gen_data_aug_config()


data_aug_config=[];

data_aug_config.do_aug=true;

data_aug_config.aug_scales=[0.7:0.1:1.3];
data_aug_config.aug_flips=[false true];


end


function data_crop_config=gen_data_crop_config(run_config)

data_crop_config=[];

data_crop_config.do_crop=true;

% data_crop_config.gen_crop_point_type='random';
data_crop_config.gen_crop_point_type='class_sample';
data_crop_config.crop_box_step_ratio=0.2;
data_crop_config.crop_box_size=run_config.crop_box_size;



end





function path_config=gen_inputpath_config_resnet_custom(train_opts, init_resnet_layer, img_scale, output_level_num, start_stage_idx)

    
    path_config=[];
    path_config.path_type='resnet';
    path_config.init_feat_net_name='resnet';
    path_config.reduce_layer_stride_topk=0;
    path_config.lr_factor=train_opts.input_net_lr_multiplier;


    input_adapt_dims=[512 256 256 256];

    path_config.input_img_scale=img_scale;
    path_config.init_output_level_idx=start_stage_idx;
    
    
    refine_config_levels=cell(output_level_num, 1);
    for o_idx=1:output_level_num

    	out_level_idx=start_stage_idx+o_idx-1;

    	path_str=sprintf('p_ims%.1f_outl%d', img_scale, out_level_idx);
		path_str(strfind(path_str, '.'))='d';

        one_refine_config=[];
        one_refine_config.input_adapt_dim=input_adapt_dims(o_idx);
        one_refine_config.input_name=path_str;
        refine_config_levels{o_idx}=one_refine_config;
    end
    
    
    path_config.output_level_num=output_level_num;
    path_config.refine_config_levels=refine_config_levels;
    
    path_config.init_resnet_layer=init_resnet_layer;
    path_config.pre_trained_model_file=sprintf('../model_trained/imagenet-resnet-%d-dag.mat', init_resnet_layer);
        
end







function path_config=gen_inputpath_config_imgraw()
    
    path_config=[];
    path_config.path_type='imgraw';
    path_config.conv_block_num=1;
    path_config.conv_num_one_block=2;
    path_config.filter_num_blocks=[64 64];
    path_config.output_dim=path_config.filter_num_blocks(end);
    path_config.input_img_scale=1;
    
    one_refine_config=[];
    one_refine_config.input_adapt_dim=path_config.filter_num_blocks(end);
    one_refine_config.input_name=sprintf('pathraw_out%d', 1);
    path_config.refine_config_levels{1}=one_refine_config;
    
    path_config.lr_factor=1;
            
end






function refine_config=gen_refine_config_stage()


	refine_config=[];
    
    refine_config.lr_factor=1;
    
    refine_config.use_chained_pool=true;
    refine_config.chained_pool_num=2;
    refine_config.chained_pool_size=5;
    
    if ~refine_config.use_chained_pool
        refine_config.chained_pool_num=0;
    end
    
    refine_config.refine_type='stage';
    refine_config.refine_block_conv_num_mainflow=3;
    refine_config.adapt_conv_num=2;

    refine_config.use_group_stage=true;
    refine_config.group_size=[];
    refine_config.group_ids=[1 2 3 4];
    
    % 2-scale setting:
%     refine_config.group_ids=[1 2 3 4 1 2 3 4];

end





function train_opts=do_config_refinenet(train_opts, class_info)


train_opts.input_net_lr_multiplier=0.1;

train_opts.bnorm_moment_type_trn='global';
train_opts.bnorm_moment_type_tst='global';

  
inputpath_configs=cell(0);


inputpath_configs{end+1, 1}=gen_inputpath_config_resnet_custom(train_opts, 50, 1.2, 4, 1);
% inputpath_configs{end+1, 1}=gen_inputpath_config_resnet_custom(train_opts, 101, 1.2, 4, 1);
% inputpath_configs{end+1, 1}=gen_inputpath_config_resnet_custom(train_opts, 152, 1.2, 4, 1);


% 2-scale setting:
% inputpath_configs{end+1, 1}=gen_inputpath_config_resnet_custom(train_opts, 50, 0.6, 4, 1);
% inputpath_configs{end+1, 1}=gen_inputpath_config_resnet_custom(train_opts, 50, 1.2, 4, 1);


% inputpath_configs{end+1, 1}=gen_inputpath_config_imgraw();


refine_config=gen_refine_config_stage();



refine_config_levels=cell(0);
path_num=length(inputpath_configs);
net_input_img_scales=zeros(path_num, 1);
for p_idx=1:path_num
    
    one_path_config=inputpath_configs{p_idx};
    one_path_config.path_idx=p_idx;
    inputpath_configs{p_idx}=one_path_config;
    
    net_input_img_scales(p_idx)=one_path_config.input_img_scale;
    refine_config_levels=cat(1, refine_config_levels, one_path_config.refine_config_levels);
end
train_opts.net_input_img_scales=net_input_img_scales;
refine_config.refine_config_levels=refine_config_levels;


loss_config.lr_factor=1;
loss_config.lossgroup_conv_num=0;
loss_config.class_num=class_info.class_num;
loss_config.loss_use_spatial_sample=false;


train_opts.refine_config=refine_config;
train_opts.inputpath_configs=inputpath_configs;
train_opts.loss_config=loss_config;


end


