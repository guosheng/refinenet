

function batch_ds_info=my_get_batch_ds_info(...
    opts, imdb, work_info, net_config, work_info_epoch, work_info_batch)


task_idx=work_info_batch.ref.task_idxes;
task_subidx=work_info_batch.ref.task_subidxes;

% only support using one image for one batch...
batch_img_num=length(task_subidx);
assert(batch_img_num==1);
assert(work_info_batch.ref.batch_task_num==1);


batch_data=get_batch_data(opts, imdb, task_idx);
img_size_input=size(batch_data.img_data);
img_size_input=img_size_input(1:2);
assert(size(batch_data.img_data, 4)==batch_img_num);

% do augmentation before crop


img_size_aug=[];

if work_info.ref.run_trn
    
    batch_data=batch_do_data_augmentation(opts, imdb, work_info_epoch, batch_data, work_info_batch, work_info);
    img_size_aug=size(batch_data.img_data);
    img_size_aug=img_size_aug(1:2);

    batch_data=batch_do_data_crop(opts, imdb, work_info_epoch, batch_data, work_info_batch, work_info);
    
else
    
    batch_data.aug_info=[];
    batch_data.crop_info=[];
end

net_input_img_scales=net_config.ref.net_input_img_scales;

[net_input_info, net_input_str]=gen_net_input_info(opts, imdb, batch_data, net_input_img_scales);
img_size=size(batch_data.img_data);
img_size=img_size(1:2);


batch_ds_info=[];
batch_ds_info.aug_info=batch_data.aug_info;
batch_ds_info.crop_info=batch_data.crop_info;
batch_ds_info.task_idx=task_idx;
batch_ds_info.img_size=img_size;
batch_ds_info.img_size_aug=img_size_aug;
batch_ds_info.img_size_input=img_size_input;
batch_ds_info.net_input_str=net_input_str;

batch_ds_info.net_input_info=net_input_info;
batch_ds_info.batch_data=batch_data;


end





function img_data=pre_process_img_net_input(imdb, img_data)

    data_norm_info=imdb.ref.data_norm_info;
           
    img_data=single(img_data);
    img_data=img_data-data_norm_info.image_mean;

end




function [net_input_info,net_input_str]=gen_net_input_info(opts, imdb, batch_data, net_input_img_scales)


feat_scale_num=length(net_input_img_scales);

org_img_size=size(batch_data.img_data);
org_img_size=org_img_size(1:2);

   

input_info_child_groups=cell(feat_scale_num, 1);
for s_idx=1:length(net_input_img_scales)
    one_scale=net_input_img_scales(s_idx);
                
    if one_scale==1
        one_img_data_scaled=batch_data.img_data;
    else
        scaled_img_size=round(org_img_size.*one_scale);
        one_img_data_scaled=imresize(batch_data.img_data, scaled_img_size);
    end

    one_child_input_info=[];
    one_child_input_info.is_group_data=false;
    one_child_input_info.x=one_img_data_scaled;
    input_info_child_groups{s_idx}=one_child_input_info;
end



for tmp_img_idx=1:length(input_info_child_groups)
    
    one_img=input_info_child_groups{tmp_img_idx}.x;
    assert(size(one_img, 4)==1);
    assert(isa(one_img, 'uint8'));
    one_img=pre_process_img_net_input(imdb, one_img);
    assert(isa(one_img, 'single'));
    input_info_child_groups{tmp_img_idx}.x=one_img;
end


net_input_info=[];
net_input_info.is_group_data=true;
net_input_info.data_child_groups=input_info_child_groups;


net_input_str=[];
for s_idx=1:feat_scale_num
    one_net_input_size=size(input_info_child_groups{s_idx}.x);
    if length(one_net_input_size)<4
        one_net_input_size(4)=1;
    end
    one_net_input_str=my_gen_array_str(one_net_input_size);
    net_input_str=cat(2, net_input_str, one_net_input_str);
end
   


net_input_info=my_init_input_info(net_input_info);


end





function batch_data=get_batch_data(train_opts, imdb, img_idx)
    
    batch_data=[];
    if train_opts.cache_data_mem
                
        if ~isfield(imdb.ref, 'data_cache_tasks') || isempty(imdb.ref.data_cache_tasks)
            imdb.ref.data_cache_tasks=cell(imdb.ref.max_task_idx, 1);
        end

        batch_data=imdb.ref.data_cache_tasks{img_idx};
        if isempty(batch_data)
            do_ds_cache_all_before_run(train_opts, imdb);
            batch_data=imdb.ref.data_cache_tasks{img_idx};
            assert(~isempty(batch_data));
        end
    end
            
    if isempty(batch_data)
        batch_data=do_load_and_cache_batch_data(train_opts, imdb, img_idx);
    end

end

function do_ds_cache_all_before_run(train_opts, imdb)

    disp('cache all data in memory ...');
    task_idxes=imdb.ref.task_idxes;
    task_num=length(task_idxes);
    for t_idx=1:task_num
        one_task_idx=task_idxes(t_idx);
        do_load_and_cache_batch_data(train_opts, imdb, one_task_idx);
        if mod(t_idx, 200)==1 || t_idx==task_num
            fprintf('caching item:%d/%d\n', t_idx, task_num)
        end
    end
    
end


function batch_data=do_load_and_cache_batch_data(train_opts, imdb, img_idx)
                        
    ds_info=imdb.ref.ds_info;
    img_data_batch=load_img_from_ds_info(ds_info, img_idx);
    assert(isa(img_data_batch, 'uint8'));
    
    img_size=[size(img_data_batch, 1), size(img_data_batch, 2)];
    img_size_input=img_size;
    
  
    if ~train_opts.use_dummy_gt
        mask_data_info=ds_info.class_idxes_mask_data_info;
        one_mask=load_mask_from_ds_info(mask_data_info, img_idx);
    else
    	one_mask=ones(img_size, 'uint8');
    end
    assert(all(size(one_mask)==img_size_input));
    
        
    todo_scale=1;
    short_edge=min(img_size);
    if short_edge<train_opts.input_img_short_edge_min
        todo_scale=train_opts.input_img_short_edge_min/short_edge;
        assert(todo_scale>=1);
    end
    
    if short_edge>train_opts.input_img_short_edge_max
        todo_scale=train_opts.input_img_short_edge_max/short_edge;
        assert(todo_scale<=1);
    end
    todo_scale=todo_scale*train_opts.input_img_scale;
            
    if todo_scale~=1
        img_data_batch=imresize(img_data_batch, todo_scale);
%         img_size=[size(img_data_batch, 1), size(img_data_batch, 2)];
    end

   
    batch_data=[];
    batch_data.img_data=img_data_batch;
    batch_data.label_data=one_mask;
    batch_data.img_idx=img_idx;
    batch_data.img_size_origin=img_size_input;
           

end




