


function disp_evaluate_result(result_info, class_info)

fprintf('\n\nClass info:\n');
disp(class_info);

fprintf('\n\nCategory result:\n');
class_num=class_info.class_num;
class_label_values=class_info.class_label_values;

for c_idx=1:class_num

	one_label_value=class_label_values(c_idx);
    if ismember(one_label_value, class_info.void_label_values)
        continue;
    end
        
    class_name=class_info.class_names{c_idx};
    
    one_iou=NaN;
    one_acc=NaN;
    
    one_iou=result_info.inter_union_score_classes(c_idx);
    
    if isfield(result_info, 'accuracy_classes')
        one_acc=result_info.accuracy_classes(c_idx);
    end
        
    fprintf('class_idx:%d, label_vale:%d, class_name:%s, pixel_acc:%.4f, IoU:%.4f\n', c_idx, one_label_value, class_name, one_acc, one_iou);
end


fprintf('\n\nOverall evaluate result:\n');
disp(result_info);

end
