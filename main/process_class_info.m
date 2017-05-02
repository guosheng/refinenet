

function class_info=process_class_info(class_info)

class_label_values=class_info.class_label_values;
assert(isa(class_label_values, 'uint8'));
assert(isa(class_info.background_label_value, 'uint8'));
assert(isa(class_info.void_label_values, 'uint8'));


class_info.background_class_idx=find(class_label_values==class_info.background_label_value, 1);
class_info.void_class_idxes=find(ismember(class_label_values, class_info.void_label_values));

class_num=length(class_info.class_names);
assert(class_num==numel(unique(class_label_values)));
class_info.class_num=class_num;

end
