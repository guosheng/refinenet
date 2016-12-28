

function ds_info=process_ds_info_classification(ds_info, ds_config)


class_info=ds_info.class_info;

img_num=length(ds_info.img_names);
class_idxes_mask_dir=fullfile(ds_info.ds_dir, 'my_class_idxes_mask');
mkdir_notexist(class_idxes_mask_dir);

fprintf('generating new masks encoded by class indexes...\n');

class_label_values=class_info.class_label_values;
assert(isa(class_label_values, 'uint8'));

class_num=length(class_label_values);
assert(class_num<2^8);

class_label_values_imgs=cell(img_num, 1);
class_idxes_imgs=cell(img_num, 1);
mask_files=cell(img_num, 1);
pixel_count_classes=zeros(class_num, 1);

assert(~ds_config.use_dummy_gt)
mask_cmap = VOClabelcolormap(256);


for img_idx=1:img_num
    
    mask_data=load_mask_from_ds_info(ds_info, img_idx);

    one_class_label_values=unique(mask_data);
%     assert(all(ismember(one_class_label_values, class_label_values)));
    class_label_values_imgs{img_idx}=one_class_label_values;

    class_idxes_mask_data=zeros(size(mask_data), 'uint8');
    tmp_class_exist_flags=false(class_num, 1);

    for tmp_idx=1:length(one_class_label_values)

        one_label_value=one_class_label_values(tmp_idx);
        one_class_idx=find(one_label_value==class_label_values, 1);
        assert(~isempty(one_class_idx));

        tmp_sel=mask_data==one_label_value;
        class_idxes_mask_data(tmp_sel)=one_class_idx;

        pixel_count_classes(one_class_idx)=pixel_count_classes(one_class_idx)+nnz(tmp_sel);

        tmp_class_exist_flags(one_class_idx)=true;

    end

    class_idxes_imgs{img_idx}=find(tmp_class_exist_flags);


    mask_file_name=ds_info.img_files{img_idx};
    [~, mask_file_name]=fileparts(mask_file_name);
       
    
    
    % save as mat files:
%     new_mask_file_short=['img_idx_' num2str(img_idx) '_' mask_file_name '.mat'];
%     new_mask_file=fullfile(class_idxes_mask_dir, new_mask_file_short);
%     save(new_mask_file, 'class_idxes_mask_data');
    
    % save as png images:
    new_mask_file_short=['img_idx_' num2str(img_idx) '_' mask_file_name '.png'];
    new_mask_file=fullfile(class_idxes_mask_dir, new_mask_file_short);
    imwrite(class_idxes_mask_data, mask_cmap, new_mask_file);
    
    if mod(img_idx, 100)==1
        fprintf('save class_idxes_mask_data(%d/%d), file:%s\n', img_idx, img_num, new_mask_file);
    end
    mask_files{img_idx}=new_mask_file_short;

end



class_idxes_mask_data_info=[];
class_idxes_mask_data_info.mask_files=mask_files;
class_idxes_mask_data_info.data_dirs={class_idxes_mask_dir};
class_idxes_mask_data_info.data_dir_idxes_mask=ones(img_num, 1, 'uint8');


ds_info.class_idxes_mask_data_info=class_idxes_mask_data_info;
ds_info.class_idxes_imgs=class_idxes_imgs;

% ds_info.class_label_values_imgs=class_label_values_imgs;
% ds_info.pixel_count_classes=pixel_count_classes;

end





