


function result_info=evaluate_predict_results(result_evaluate_param, class_info)

gt_mask_dir=result_evaluate_param.gt_mask_dir;
predict_result_dir=result_evaluate_param.predict_result_dir;
my_check_file(gt_mask_dir);
my_check_file(predict_result_dir);

result_info=do_eva_conmat(gt_mask_dir, predict_result_dir, class_info);


do_verify=false;
% do_verify=true;

if do_verify
    % for verifying the evaluation scores with another toolkit
    result_info.result_info_ADE_toolkit=do_eva_ADE_toolkit(gt_mask_dir, predict_result_dir, class_info);

    disp('result_info:')
    disp(result_info);

    disp('result_info.result_info_ADE_toolkit');
    disp(result_info.result_info_ADE_toolkit);
end

end



function new_mask=do_transfer_mask(mask, class_info)

% transfer the label value mask to class idx mask

class_label_values=class_info.class_label_values;

new_mask=zeros(size(mask), 'uint8');
can_values=unique(mask);
assert(length(can_values)<255);

for can_idx=1:length(can_values)
    one_label_v=can_values(can_idx);
    class_idx=find(one_label_v==class_label_values);
    assert(length(class_idx)==1);
    tmp_flags=mask==one_label_v;
    new_mask(tmp_flags)=uint8(class_idx);
end

end


function result_info=do_eva_conmat(gt_dir, predict_dir, class_info)


fileinfos=dir(fullfile(predict_dir, '*.png'));
test_num=length(fileinfos);
assert(test_num>0)

exclude_class_idxes=class_info.void_class_idxes;
class_num=class_info.class_num;
pixel_con_mat=zeros(class_num, class_num);
    
for i = 1:test_num
    file_name=fileinfos(i).name;
    gt_file=fullfile(gt_dir, file_name);
    predict_file=fullfile(predict_dir, file_name);
    gt_mask=imread(gt_file);
    predict_mask=imread(predict_file);
    
    gt_mask=do_transfer_mask(gt_mask, class_info);
    predict_mask=do_transfer_mask(predict_mask, class_info);
    
    one_result=seg_eva_one_img(predict_mask, gt_mask, class_info);
    pixel_con_mat=pixel_con_mat+one_result.confusion_mat;
    
    if mod(i, 20)==0
        fprintf('evaluating: %d/%d\n', i, test_num);
    end
end

result_info=seg_eva_gen_result_from_con_mat(pixel_con_mat, exclude_class_idxes);

end





function result_info=do_eva_ADE_toolkit(gt_dir, predict_dir, class_info)

% This function takes the prediction and label of a single image, returns intersection and union areas for each class
% To compute over many images do:

fileinfos=dir(fullfile(predict_dir, '*.png'));
test_num=length(fileinfos);
assert(test_num>0)

void_class_flags=ismember(class_info.class_label_values, class_info.void_label_values);
valid_class_num=nnz(~void_class_flags);

need_transfer_label=false;

if any(void_class_flags)
    assert(nnz(void_class_flags)==1);

    assert(void_class_flags(end) || void_class_flags(1));

    if void_class_flags(end)
        % in one case, the void label is 255, void class idx should be the last class label
        need_transfer_label=true;        	
    end

    if void_class_flags(1)
        % in another case, the void label is 0, void class idx should be the first class label
        assert(class_info.void_label_values==0);
    end
    
end

if ~need_transfer_label
    % in this case, class labels should be in a regular form
	valid_class_labels=class_info.class_label_values(~void_class_flags);
	expected_label_values=(1:length(valid_class_labels))';
	assert(all(valid_class_labels(:)==expected_label_values));
end


appeared_labels_in_gt=[];

for i = 1:test_num
    file_name=fileinfos(i).name;
    gt_file=fullfile(gt_dir, file_name);
    predict_file=fullfile(predict_dir, file_name);
    gt_mask=imread(gt_file);
    predict_mask=imread(predict_file);
    
    appeared_labels_in_gt=unique(cat(1, appeared_labels_in_gt, unique(gt_mask)));
    
    if need_transfer_label
	    gt_mask=do_transfer_mask(gt_mask, class_info);
	    gt_mask=handle_void_class_ADE_toolkit(gt_mask, class_info);
	    
	    predict_mask=do_transfer_mask(predict_mask, class_info);
	    predict_mask=handle_void_class_ADE_toolkit(predict_mask, class_info);
	end
    
    [area_intersection(:,i), area_union(:,i)]=intersectionAndUnion_ADE_toolkit(predict_mask, gt_mask, valid_class_num);
    
    if mod(i, 20)==0
        fprintf('evaluating: %d/%d\n', i, test_num);
    end
end
IoU = sum(area_intersection,2)./sum(eps+area_union,2);

appeared_valid_labels_in_gt=appeared_labels_in_gt(~ismember(appeared_labels_in_gt, class_info.void_label_values));
result_info.appeared_valid_labels_in_gt=appeared_valid_labels_in_gt;
result_info.appeared_valid_label_num_in_gt=length(result_info.appeared_valid_labels_in_gt);
result_info.appeared_label_iou_mean=sum(IoU)./result_info.appeared_valid_label_num_in_gt;

% update the vliad_class_num, only count those appeared in groundtruth masks.
result_info.original_valid_class_num=valid_class_num;
valid_class_num=result_info.appeared_valid_label_num_in_gt;

result_info.valid_class_num=valid_class_num;
result_info.inter_union_score_classes=IoU;
result_info.inter_union_score_mean=sum(IoU)./valid_class_num;


end


function new_mask=handle_void_class_ADE_toolkit(new_mask, class_info)

ignore_class_idx=class_info.void_class_idxes;

% in ADE toolkit, assume the void label value is 0:
if ~isempty(ignore_class_idx)
    assert(length(ignore_class_idx)==1);
    new_mask(new_mask==ignore_class_idx)=0;
end

end



%copy form ADE20K tookit:

function [area_intersection, area_union] = intersectionAndUnion_ADE_toolkit(imPred, imLab, numClass)
% This function takes the prediction and label of a single image, returns intersection and union areas for each class
% To compute over many images do:
% for i = 1:Nimages
%  [area_intersection(:,i), area_union(:,i)]=intersectionAndUnion(imPred{i}, imLab{i});
% end
% IoU = sum(area_intersection,2)./sum(eps+area_union,2);

imPred = uint16(imPred(:));
imLab = uint16(imLab(:));

% Remove classes from unlabeled pixels in label image. 
% We should not penalize detections in unlabeled portions of the image.
imPred = imPred.*uint16(imLab>0);  

% Compute area intersection
intersection = imPred.*uint16(imPred==imLab);
area_intersection = hist(intersection, 0:numClass);

% Compute area union
area_pred = hist(imPred, 0:numClass);
area_lab = hist(imLab, 0:numClass);
area_union = area_pred + area_lab - area_intersection;

% Remove unlabeled bin and convert to uint64
area_intersection = area_intersection(2:end);
area_union = area_union(2:end);

end


