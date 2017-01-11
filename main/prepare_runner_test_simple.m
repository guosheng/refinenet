
function runner_info=prepare_runner_test_simple(run_config)

class_info=run_config.class_info;

ds_config=[];


% set the input image scales, useful for multi-scale evaluation
run_config.input_img_scale=0.8;
run_config.gen_net_opts_fn=@gen_net_opts_model_type1;


run_config.run_evaonly=true;
ds_config.use_custom_data=true;
ds_config.use_dummy_gt=true;
run_config.use_dummy_gt=ds_config.use_dummy_gt;


ds_config.ds_name='tmp_data';
run_config.root_cache_dir=fullfile('../cache_data', 'tmp_runner_dir');
% mkdir_notexist(run_config.root_cache_dir);

run_config.model_name='tmp_model_name';

run_dir_name=fileparts(mfilename('fullpath'));
[~, run_dir_name]=fileparts(run_dir_name);
run_config.run_dir_name=run_dir_name;
run_config.run_file_name=mfilename();


train_opts=run_config.gen_net_opts_fn(run_config, class_info);

net_run_opts=train_opts.net_run_opts;
net_run_opts.batch_info_disp_fn=[];
net_run_opts.epoch_evaluate_fn=@(work_info, opts, imdb, net_config, work_info_epoch){};
train_opts.net_run_opts=net_run_opts;

ds_info=[];
ds_info.train_idxes=1;
ds_info.test_idxes=1;
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

loss_group_info=net_config.ref.group_infos{end};
loss_group_info.forward_evaluate_fn=@tmp_loss_group_evaluate_runner;
net_config.ref.group_infos{end}=loss_group_info;

runner_info=[];
runner_info.train_opts=train_opts;
runner_info.imdb=imdb;
runner_info.net_config=net_config;
runner_info.net_exp_info=net_exp_info;
runner_info.class_info=class_info;
runner_info.ds_config=ds_config;

runner_info.run_task_fn=@run_task;

end




function ds_info=gen_runner_ds_info(ds_config, class_info, task_info)

    ds_info=[];

    img_dir=task_info.img_dir;
    img_files={task_info.img_filename};
    img_num=1;
    ds_info.img_idxes=uint32(1:img_num);
    ds_info.img_num=img_num;
    ds_info.train_idxes=uint32([]);
    ds_info.test_idxes=ds_info.img_idxes;
    
    img_names=cell(img_num, 1);
    for img_idx=1:img_num
        one_name=img_files{img_idx};
        [~, one_name]=fileparts(one_name);
        img_names{img_idx}=one_name;
    end
        
        
    ds_info.img_files=img_files;
    ds_info.img_names=img_names;
    ds_info.mask_files=[];
    ds_info.train_idxes=[];
    ds_info.test_idxes=uint32(1:img_num)';
    ds_info.name=ds_config.ds_name;
    
    ds_info.data_dir_idxes_img=repmat(uint8(1), [img_num, 1]);
    ds_info.data_dir_idxes_mask=[];
    ds_info.data_dirs={img_dir};
    
    ds_info.class_info=class_info;
    
end



function task_result=run_task(runner_info, task_info)

runner_info.imdb.ref.ds_info=gen_runner_ds_info(runner_info.ds_config, runner_info.class_info, task_info);
exp_info=my_net_tool(runner_info.train_opts, runner_info.imdb, runner_info.net_config, runner_info.net_exp_info);
work_info=exp_info.eva_val;

task_result=work_info.ref.task_result_imgs{1};

end









function tmp_loss_group_evaluate_runner(work_info_batch, group_idx)
       

    cnn_forward_evaluate_obj(work_info_batch, group_idx);
    cnn_softmax_evaluate(work_info_batch, group_idx);
    
    if ~work_info_batch.ref.run_eva
        return;
    end
           
    predict_info=gen_predict_info(work_info_batch, group_idx);
    do_full_eva(work_info_batch, predict_info);
    
end


function predict_info=gen_predict_info(work_info_batch, group_idx)
                
    prediction_info=work_info_batch.ref.prediction_info_groups{group_idx};
    one_output_info=prediction_info.output_infos{end};
    assert(check_valid_net_output(one_output_info));
    predict_info=one_output_info.mc_predict_info;
    
end


function do_full_eva(work_info_batch, predict_info)
    
    work_info=work_info_batch.ref.work_info;
    opts=work_info_batch.ref.train_opts;
        
    %assume no crop or data_aug
    batch_ds_info=work_info_batch.ref.ds_info;
    assert(isempty(batch_ds_info.aug_info));
    assert(isempty(batch_ds_info.crop_info));
        
    task_subidx=work_info_batch.ref.task_subidxes;
    img_idx=work_info_batch.ref.task_idxes;
    img_num=length(task_subidx);
    assert(img_num==1);
       
    
    seg_param=[];
    seg_param.eva_param=opts.eva_param;
    seg_param.img_idx=img_idx;
    
    batch_data=batch_ds_info.batch_data;
    seg_param.img_data_input=batch_data.img_data;
    seg_param.gt_mask_input=batch_data.label_data;
    seg_param.img_size_input=batch_data.img_size_origin;
    
    ds_info=work_info_batch.ref.imdb.ref.ds_info;
    result_info=do_gen_seg_result(seg_param, ds_info, predict_info);
        
    work_info.ref.task_result_imgs{task_subidx}=result_info;
           
    
end





function result_info=do_gen_seg_result(seg_param, ds_info, predict_info)
    
    
    
    img_size=seg_param.img_size_input;
       
   
    assert(~isempty(predict_info));
    
    score_map=single(predict_info.score_map);
    score_map_size=size(score_map);
    score_map_size=score_map_size(1:2);
    
    score_map_org=score_map;
        
    if any(img_size~=score_map_size)
        score_map=log(score_map);
        score_map=max(score_map, -20);
        score_map=my_resize(score_map, img_size);
        score_map=exp(score_map);
    end
       

    [~, predict_mask]=max(score_map,[],3);
    predict_mask=uint8(gather(predict_mask));
              
                
    
    result_info=gen_task_result(ds_info, seg_param, predict_mask, score_map_org);
          
end



function tmp_result_info=gen_task_result(ds_info, seg_param, predict_mask_net, score_map_org)

    img_idx=seg_param.img_idx;
    eva_param=seg_param.eva_param;
    class_info=eva_param.class_info;
    
    predict_mask_data=class_info.class_label_values(predict_mask_net);      
    assert(isa(class_info.class_label_values, 'uint8'));	
    assert(isa(predict_mask_data, 'uint8'));
    
    img_name=ds_info.img_names{img_idx};
               
    tmp_result_info=[];
    tmp_result_info.mask_data=predict_mask_data;
    tmp_result_info.img_size=size(predict_mask_data);
    tmp_result_info.class_info=class_info;
    tmp_result_info.img_name=img_name;
    tmp_result_info.score_map=im2uint8(score_map_org);
     
    
end



