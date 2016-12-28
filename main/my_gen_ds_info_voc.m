


function ds_info=my_gen_ds_info_voc(ds_config)



ds_dir=fullfile('../datasets', 'voc2012_trainval');

train_idx_file=fullfile(ds_dir, 'ImageSets/Segmentation/train.txt');
fid=fopen(train_idx_file);
train_file_names=textscan(fid, '%s');
train_file_names=train_file_names{1};
fclose(fid);


val_idx_file=fullfile(ds_dir, 'ImageSets/Segmentation/val.txt');
fid=fopen(val_idx_file);
val_file_names=textscan(fid, '%s');
val_file_names=val_file_names{1};
fclose(fid);


train_num=length(train_file_names);
img_names=cat(1, train_file_names, val_file_names);
img_num=length(img_names);

img_files=cell(img_num, 1);
mask_files=cell(img_num, 1);

for t_idx=1:img_num
    file_name=img_names{t_idx};
    mask_files{t_idx}=[file_name '.png'];
    img_files{t_idx}=[file_name '.jpg'];
end

train_idxes=1:train_num;
val_idxes=train_num+1:img_num;


ds_info=[];

ds_info.img_names=img_names;
ds_info.img_files=img_files;
ds_info.mask_files=mask_files;

ds_info.train_idxes=uint32(train_idxes');
ds_info.test_idxes=uint32(val_idxes');


img_dir=fullfile(ds_dir, 'JPEGImages');
mask_dir=fullfile(ds_dir, 'SegmentationClass');


data_dirs=[];
data_dirs{1}=img_dir;
data_dirs{2}=mask_dir;

data_dir_idxes_img=zeros([img_num 1], 'uint8')+1;
data_dir_idxes_mask=zeros([img_num 1], 'uint8')+2;

ds_info.data_dir_idxes_img=data_dir_idxes_img;
ds_info.data_dir_idxes_mask=data_dir_idxes_mask;
ds_info.data_dirs=data_dirs;
ds_info.ds_dir=ds_dir;

ds_info.class_info=gen_class_info_voc();

ds_info.ds_name='voc2012_trainval';

ds_info=process_ds_info_classification(ds_info, ds_config);

end


