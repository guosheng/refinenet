

function ds_info=gen_dataset_info(ds_config)

if ds_config.use_custom_data
    
    ds_info=gen_ds_info_custom_data(ds_config, ds_config.class_info);
    
else
    
    cache_dir=ds_config.ds_info_cache_dir;
    cache_file=fullfile(cache_dir, 'my_dataset_info.mat');

    if ~my_check_file(cache_file)
        ds_info=ds_config.gen_ds_info_fn(ds_config);
        my_save_file(cache_file, ds_info);
    else
        ds_info=my_load_file(cache_file);
        ds_info=ds_info.data_obj;
    end
    
end
  

end


function ds_info=gen_ds_info_custom_data(ds_config, class_info)

    ds_info=[];

    img_dir=ds_config.img_data_dir;
    img_files=my_list_file(img_dir)';
    img_num=length(img_files);
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






