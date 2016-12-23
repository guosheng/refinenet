
function class_info=gen_class_info_sunrgbd()

class_info=[];

class_names=load('./sunrgbd_class_names.mat');
class_names=class_names.seg37list;
class_num_org=length(class_names);

class_names = cat(1, {'void'}, class_names'); 
class_info.class_names=class_names;

 
class_label_values=uint8([0 1:class_num_org]);
class_info.class_label_values=class_label_values;

class_info.background_label_value=uint8(1);
class_info.void_label_values=uint8(0);

class_info.mask_cmap = VOClabelcolormap(256);
class_info=process_class_info(class_info);


end


