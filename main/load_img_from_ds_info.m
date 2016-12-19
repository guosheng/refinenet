


function img_data=load_img_from_ds_info(img_data_info, img_idx)

assert(~isempty(img_idx));

img_file=img_data_info.img_files{img_idx};
one_data_dir_idx=img_data_info.data_dir_idxes_img(img_idx);
one_data_dir=img_data_info.data_dirs{one_data_dir_idx};

full_img_file=fullfile(one_data_dir, img_file);
img_data=read_img_rgb(full_img_file, 'uint8');

end