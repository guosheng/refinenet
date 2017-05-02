
function class_info=gen_class_info_ade()

class_info_ADE=load('class_info_ADE.mat');
class_info_ADE=class_info_ADE.class_info_ADE;

class_names=class_info_ADE.Name;
assert(size(class_names, 2)==1);
class_names=cat(1, {'void'}, class_names);

class_label_values=uint8([0 1:150]);



class_info=[];

class_info.class_names = class_names;


class_info.class_label_values=class_label_values;
class_info.background_label_value=uint8(1);
class_info.void_label_values=uint8(0);

% addpath ../libs/VOCdevkit_2012/VOCcode
class_info.mask_cmap = VOClabelcolormap(256);

class_info=process_class_info(class_info);

end
