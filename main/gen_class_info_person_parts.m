
function class_info=gen_class_info_person_parts()

class_info=[];

class_info.class_names={'background'; 'Head'; 'Torso'; 'Upper Arms'; 'Lower Arms'; 'Upper Legs'; 'Lower Legs'};
assert(length(class_info.class_names)==7);
class_label_values=uint8([0:6]);
        

class_info.class_label_values=class_label_values;
class_info.background_label_value=uint8(0);
class_info.void_label_values=uint8(255);

class_info.mask_cmap = VOClabelcolormap(256);

class_info=process_class_info(class_info);

end