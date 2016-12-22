
function class_info=gen_class_info_cityscapes()

class_info=[];

class_info.class_names={'road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'trafficlight',...
     'trafficsign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car', ...
     'truck', 'bus', 'train', 'motorcycle', 'bicycle', 'void'};

 
class_label_values=uint8([0:18 255]);
       

class_info.class_label_values=class_label_values;
class_info.background_label_value=uint8(0);
class_info.void_label_values=uint8(255);

cmap=load('cityscape_cmap.mat');
cmap=uint8(cmap.cityscape_cmap);
class_info.mask_cmap=im2double(cmap);

class_info=process_class_info(class_info);

end


