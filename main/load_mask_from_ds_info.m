


function mask_data=load_mask_from_ds_info(mask_data_info, img_idx)


    mask_file=mask_data_info.mask_files{img_idx};
    assert(~isempty(mask_file));
   
    one_data_dir_idx=mask_data_info.data_dir_idxes_mask(img_idx);
    one_data_dir=mask_data_info.data_dirs{one_data_dir_idx};
    
    full_mask_file=fullfile(one_data_dir, mask_file);
        
    [~, ~, fileext]=fileparts(mask_file);
    if strcmp(fileext, '.mat')
        mask_data = my_load_file(full_mask_file, true);
        mask_data=mask_data.class_idx_mask_data;
    else
        mask_data=imread(full_mask_file);
        assert(isa(mask_data, 'uint8'));
    end
    
end


