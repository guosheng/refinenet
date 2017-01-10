
function class_info=gen_class_info_nyud()

class_info=[];

class_info.class_names={
    'wall'
    'floor'
    'cabinet'
    'bed'
    'chair'
    'sofa'
    'table'
    'door'
    'window'
    'bookshelf'
    'picture'
    'counter'
    'blinds'
    'desk'
    'shelves'
    'curtain'
    'dresser'
    'pillow'
    'mirror'
    'floor mat'
    'clothes'
    'ceiling'
    'books'
    'refridgerator'
    'television'
    'paper'
    'towel'
    'shower curtain'
    'box'
    'whiteboard'
    'person'
    'night stand'
    'toilet'
    'sink'
    'lamp'
    'bathtub'
    'bag'
    'otherstructure'
    'otherfurniture'
    'otherprop'
    'void'};

 
class_label_values=uint8([1:40 255]);
class_info.class_label_values=class_label_values;

class_info.background_label_value=uint8(1);
class_info.void_label_values=uint8(255);

class_info.mask_cmap = VOClabelcolormap(256);
class_info=process_class_info(class_info);


end


