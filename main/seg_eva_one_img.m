

function one_pixel_result=seg_eva_one_img(predict_mask, gt_mask, class_info)


    exclude_class_idxes=class_info.void_class_idxes;
    class_num=class_info.class_num;
      
    assert(all(size(gt_mask)==size(predict_mask)));
        
    gt_mask_vector=gt_mask(:);
    predict_mask_vector=predict_mask(:);

    if ~isempty(exclude_class_idxes)
        valid_pixel_sel=~ismember(gt_mask, exclude_class_idxes);
        gt_mask_vector=gt_mask_vector(valid_pixel_sel);
        predict_mask_vector=predict_mask_vector(valid_pixel_sel);
    end


    [tmp_con_mat, label_vs]=confusionmat(gt_mask_vector, predict_mask_vector);
    confusion_mat=zeros(class_num, class_num);
    confusion_mat(label_vs,label_vs)=tmp_con_mat;

    one_pixel_result=seg_eva_gen_result_from_con_mat(confusion_mat, exclude_class_idxes);
            

end


