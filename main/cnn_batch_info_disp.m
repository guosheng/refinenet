

function cnn_batch_info_disp(train_opts, work_info_batch)

    batch_ds_info=work_info_batch.ref.ds_info;
    
    img_size_str=my_gen_array_str(batch_ds_info.img_size);
    img_size_str_input=my_gen_array_str(batch_ds_info.img_size_input);
    img_size_str_aug=my_gen_array_str(batch_ds_info.img_size_aug);
    
    feat_map_size_inputpaths=work_info_batch.ref.feat_map_size_inputpaths;
    inputpath_num=size(feat_map_size_inputpaths, 1);
    feat_size_str_inputpath=[];
    for p_idx=1:inputpath_num
        if isempty(feat_map_size_inputpaths)
            feat_map_size=[];
        else
            feat_map_size=feat_map_size_inputpaths{p_idx};
        end
        one_feat_size_str_inputpath=my_gen_array_str(feat_map_size);
        feat_size_str_inputpath=cat(2, feat_size_str_inputpath, one_feat_size_str_inputpath);
    end
    
    net_input_img_scales=work_info_batch.ref.net_config.ref.net_input_img_scales;
    img_scales_str=my_gen_array_str(net_input_img_scales, '%.1f');
    net_input_str=batch_ds_info.net_input_str;
        
    predict_map_size_str=my_gen_array_str(work_info_batch.ref.predict_map_size);
    valid_node_ratio=work_info_batch.ref.valid_node_ratio;
    
            
    aug_info=batch_ds_info.aug_info;
    if ~isempty(aug_info) && aug_info.aug_param.do_aug
        data_aug_config=train_opts.data_aug_config;
        aug_scales_str=my_gen_array_str(data_aug_config.aug_scales([1,end]), '%.1f');
        aug_flips_str=my_gen_array_str(data_aug_config.aug_flips);  
        one_scale_str=my_gen_array_str(aug_info.aug_param.scale, '%.1f');
    else
        aug_scales_str='[]';
        aug_flips_str='[]';
        one_scale_str='[]';
    end
        
    crop_info=batch_ds_info.crop_info;
    if ~isempty(crop_info) && crop_info.do_crop
        do_crop_str=my_gen_array_str(train_opts.data_crop_config.crop_box_size);
    else
        do_crop_str='[]';
    end
    
    
    refine_config=train_opts.refine_config;
	pool_sizes_str='[]';
    pool_num=0;
    if ~isempty(refine_config)
        if refine_config.use_chained_pool
            pool_sizes_str=my_gen_array_str(refine_config.chained_pool_size);
        end
        pool_num=refine_config.chained_pool_num;
    end
    
        
    class_info=train_opts.eva_param.class_info;
        
    
    fprintf('--batch_info, dir: %s, file: %s\n', ...
            train_opts.run_dir_name, train_opts.run_file_name) ;
        
    fprintf('--batch_info, model_name: %s\n', train_opts.model_name) ;
    
    fprintf('--batch_info, input_scale:%.1f, net_scales:%s, chained_pool:%d(size:%s), valid_node:%.2f, class:%d \n', ...
        train_opts.input_img_scale,  img_scales_str, pool_num, pool_sizes_str, valid_node_ratio, class_info.class_num) ;
            
    fprintf('--batch_info, img_size:%s(input:%s, aug:%s), data_aug:(scale:%s(range:%s), flip:%s), crop:%s\n', ...
            img_size_str, img_size_str_input, img_size_str_aug, ...
            one_scale_str, aug_scales_str, aug_flips_str, do_crop_str) ;
    
    fprintf('--batch_info, net_input:%s, feat_inputpaths:%s, net_output:%s\n', ...
            net_input_str, feat_size_str_inputpath, predict_map_size_str) ;           
 

end


