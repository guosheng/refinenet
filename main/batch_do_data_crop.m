


function batch_data=batch_do_data_crop(train_opts, imdb, work_info_epoch, batch_data, work_info_batch, work_info)


data_crop_config=train_opts.data_crop_config;

if work_info.ref.run_eva || ~data_crop_config.do_crop
    batch_data.crop_info=[];
    return;
end


img_data=batch_data.img_data;
mask_data=batch_data.label_data;
img_size=size(img_data);
img_size=img_size(1:2);

if any(size(mask_data)~=img_size)
    mask_data=imresize(mask_data, img_size, 'nearest');
end


min_edge_size=min(img_size);
need_changed_flags=mod(min_edge_size,2)~=0;
min_edge_size(need_changed_flags)=min_edge_size(need_changed_flags)-1;

crop_box_size=data_crop_config.crop_box_size;
crop_box_size=min(min_edge_size, crop_box_size);


gen_crop_point_type=data_crop_config.gen_crop_point_type;

start_point=[];

if strcmp(gen_crop_point_type, 'random')
    start_point=gen_crop_point_random(data_crop_config, imdb, work_info_batch, mask_data, crop_box_size);

elseif strcmp(gen_crop_point_type, 'class_sample')
    
    start_point=gen_crop_point_class_sample(data_crop_config, imdb, work_info_batch, mask_data, crop_box_size);
    if isempty(start_point)
        start_point=gen_crop_point_random(data_crop_config, imdb, work_info_batch, mask_data, crop_box_size);
    end

else
	
	error('gen_crop_point_type not support!');
end

assert(~isempty(start_point));

stop_point1=start_point(1)+crop_box_size-1;
if stop_point1>img_size(1)
    stop_point1=img_size(1);
    start_point(1)=stop_point1-crop_box_size+1;
end
    
stop_point2=start_point(2)+crop_box_size-1;
if stop_point2>img_size(2)
    stop_point2=img_size(2);
    start_point(2)=stop_point2-crop_box_size+1;
end

row_idxes=start_point(1):stop_point1;
col_idxes=start_point(2):stop_point2;


crop_img_data=img_data(row_idxes, col_idxes, :);
crop_mask_data=mask_data(row_idxes, col_idxes);


crop_info=[];
crop_info.img_size=img_size;
crop_info.row_idxes=row_idxes;
crop_info.col_idxes=col_idxes;
crop_info.crop_mask_data=crop_mask_data;
crop_info.crop_img_data=crop_img_data;
crop_info.mask_data=mask_data;
crop_info.img_data=img_data;
crop_info.do_crop=true;


batch_data.crop_info=crop_info;

batch_data.img_data=crop_img_data;
batch_data.label_data=crop_mask_data;

crop_img_size=size(crop_img_data);
crop_img_size=crop_img_size(1:2);
batch_data.img_size=crop_img_size;


end




function start_point=gen_crop_point_random(data_crop_config, imdb, work_info_batch, mask_data, crop_box_size)


img_size=size(mask_data);

crop_box_step_ratio=data_crop_config.crop_box_step_ratio;
if ~isempty(crop_box_step_ratio)
    step_size=round(crop_box_size*crop_box_step_ratio);    
    step_size=max(step_size, 1);
else
    step_size=1;
end

max_range=img_size-crop_box_size+1;


can_row_points=[1:step_size:max_range(1) max_range(1)];
can_col_points=[1:step_size:max_range(2) max_range(2)];
start_point=[my_random_sample(can_row_points, 1) my_random_sample(can_col_points, 1)];

end




function start_point=gen_crop_point_class_sample(data_crop_config, imdb, work_info_batch, mask_data, crop_box_size)


if ~isfield(imdb.ref, 'crop_cache_info') || isempty(imdb.ref.crop_cache_info)
    imdb.ref.crop_cache_info=gen_crop_cache_info(work_info_batch.ref.train_opts, imdb.ref.ds_info);
end


crop_cache_info=imdb.ref.crop_cache_info;
task_idx=work_info_batch.ref.task_idxes;
class_idxes=crop_cache_info.class_idxes_imgs{task_idx};

start_point=[];

if isempty(class_idxes)
    disp('################# class_sample crop: empty class_idxes!');
    return;
end


one_class_idx=my_random_sample(class_idxes, 1);
valid_flags=mask_data==one_class_idx;

valid_idxes=find(valid_flags);

if isempty(valid_idxes)
    disp('################# class_sample crop: no points with the selected class label are found!');
    return;
end

tmp_point_idx=my_random_sample(valid_idxes, 1);
[start_point(1), start_point(2)]=ind2sub(size(valid_flags), tmp_point_idx);


assert(~isempty(start_point));


start_point=round(start_point-crop_box_size./2);
start_point=max(start_point, 1);



end



function crop_cache_info=gen_crop_cache_info(train_opts, ds_info)


disp('gen_crop_cache_info...');
class_idxes_imgs=ds_info.class_idxes_imgs;

void_class_idxes=ds_info.class_info.void_class_idxes;
for img_idx=1:length(class_idxes_imgs)
    class_idxes=class_idxes_imgs{img_idx};
    assert(max(class_idxes)<256);
    if ~isempty(void_class_idxes)
        class_idxes=class_idxes(class_idxes~=void_class_idxes);
    end
    class_idxes_imgs{img_idx}=uint8(class_idxes);
end

crop_cache_info=[];
crop_cache_info.class_idxes_imgs=class_idxes_imgs;

end



