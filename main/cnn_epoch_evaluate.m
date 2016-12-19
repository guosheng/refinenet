

function cnn_epoch_evaluate(work_info, opts, imdb, net_config, work_info_epoch)


if ~work_info.ref.run_eva
    return;
end

if opts.use_dummy_gt
    return;
end


eva_param=opts.eva_param;
class_info=eva_param.class_info;

eva_result_epoch=work_info_epoch.ref.eva_result;
valid_task_flags=work_info_epoch.ref.valid_task_flags;

seg_eva_result_imgs_coarse=work_info_epoch.ref.seg_eva_result_imgs_coarse;
if ~isempty(valid_task_flags)
    seg_eva_result_imgs_coarse=seg_eva_result_imgs_coarse(valid_task_flags);
end
assert(~isempty(seg_eva_result_imgs_coarse));
seg_eva_result_ds_coarse=gen_predict_result(seg_eva_result_imgs_coarse, class_info);

eva_result_epoch.seg_accuracy_global_coarse=seg_eva_result_ds_coarse.accuracy_global;
eva_result_epoch.seg_accuracy_per_class_coarse=seg_eva_result_ds_coarse.accuracy_per_class;
eva_result_epoch.seg_inter_union_score_per_class_coarse=seg_eva_result_ds_coarse.inter_union_score_per_class;
eva_result_epoch.exclude_class_idxes_coarse=seg_eva_result_ds_coarse.exclude_class_idxes;



seg_eva_result_imgs=work_info_epoch.ref.seg_eva_result_imgs;
if ~isempty(valid_task_flags)
    seg_eva_result_imgs=seg_eva_result_imgs(valid_task_flags);
end
assert(~isempty(seg_eva_result_imgs));
seg_eva_result_ds=gen_predict_result(seg_eva_result_imgs, class_info);

eva_result_epoch.seg_accuracy_global=seg_eva_result_ds.accuracy_global;
eva_result_epoch.seg_accuracy_per_class=seg_eva_result_ds.accuracy_per_class;
eva_result_epoch.seg_inter_union_score_per_class=seg_eva_result_ds.inter_union_score_per_class;
eva_result_epoch.exclude_class_idxes=seg_eva_result_ds.exclude_class_idxes;





seg_eva_result_ds_densecrf=[];
if eva_param.eva_densecrf_postprocess

    seg_eva_result_imgs_densecrf=work_info_epoch.ref.seg_eva_result_imgs_densecrf;
    if ~isempty(valid_task_flags)
        seg_eva_result_imgs_densecrf=seg_eva_result_imgs_densecrf(valid_task_flags);
    end
    assert(~isempty(seg_eva_result_imgs_densecrf));
    seg_eva_result_ds_densecrf=gen_predict_result(seg_eva_result_imgs_densecrf, class_info);
    
    eva_result_epoch.seg_accuracy_global_densecrf=seg_eva_result_ds_densecrf.accuracy_global;
    eva_result_epoch.seg_accuracy_per_class_densecrf=seg_eva_result_ds_densecrf.accuracy_per_class;
    eva_result_epoch.seg_inter_union_score_per_class_densecrf=seg_eva_result_ds_densecrf.inter_union_score_per_class;
    eva_result_epoch.exclude_class_idxes_densecrf=seg_eva_result_ds_densecrf.exclude_class_idxes;
end





eva_names=cell(0, 1);
eva_names_disp=cell(0, 1);


% eva_names{end+1, 1}='seg_accuracy_global_coarse';
% eva_names_disp{end+1, 1}='seg accuracy global (coarse)';

% eva_names{end+1, 1}='seg_accuracy_per_class_coarse';
% eva_names_disp{end+1, 1}='seg accuracy perclass (coarse)';

eva_names{end+1, 1}='seg_inter_union_score_per_class_coarse';
eva_names_disp{end+1, 1}='IoU (coarse)';

eva_names{end+1, 1}='seg_accuracy_global';
eva_names_disp{end+1, 1}='accuracy global';

eva_names{end+1, 1}='seg_accuracy_per_class';
eva_names_disp{end+1, 1}='accuracy perclass';

eva_names{end+1, 1}='seg_inter_union_score_per_class';
eva_names_disp{end+1, 1}='IoU';


eva_result_epoch.eva_names=cat(1, eva_result_epoch.eva_names, eva_names);
eva_result_epoch.eva_names_disp=cat(1, eva_result_epoch.eva_names_disp, eva_names_disp);
    
work_info_epoch.ref.eva_result=eva_result_epoch;

fprintf('\n\n------------------------------------------------------------\n');

fprintf('\n\nevaluate result (coarse):\n');
disp(seg_eva_result_ds_coarse);

fprintf('\n\n------------------------------------------------------------\n');
disp_eva_result(seg_eva_result_ds, class_info)

fprintf('\n\nevaluate result:\n');
disp(seg_eva_result_ds);

if ~isempty(seg_eva_result_ds_densecrf)
    fprintf('\n\n------------------------------------------------------------\n');
    disp_eva_result(seg_eva_result_ds_densecrf, class_info)
    
    fprintf('\n\nevaluate result (densecrf):\n');
    disp(seg_eva_result_ds_densecrf);
    
end


end



function disp_eva_result(pixel_result, class_info)

fprintf('\n\nCategory result:\n');
class_num=class_info.class_num;
class_label_values=class_info.class_label_values;

for c_idx=1:class_num

	one_label_value=class_label_values(c_idx);
    class_name=class_info.class_names{c_idx};
    
    one_iou=pixel_result.inter_union_score_classes(c_idx);
    one_acc=pixel_result.accuracy_classes(c_idx);
        
    fprintf('class_idx:%d, label_vale:%d, class_name:%s, pixel_acc:%.4f, IoU:%.4f\n', c_idx, one_label_value, class_name, one_acc, one_iou);
end


end



function pixel_result=gen_predict_result(pixel_results, class_info)
        
    exclude_class_idxes=class_info.void_class_idxes;
    class_num=class_info.class_num;
    img_num=length(pixel_results);

    pixel_con_mat=zeros(class_num, class_num);

    for img_idx_idx=1:img_num
        one_pixel_result=pixel_results{img_idx_idx};
        pixel_con_mat=pixel_con_mat+one_pixel_result.confusion_mat;        
    end


    pixel_result=seg_eva_gen_result_from_con_mat(pixel_con_mat, exclude_class_idxes);

    predict_result=[];
    predict_result.class_num=class_num;
    predict_result.pixel_result=pixel_result;
    predict_result.exclude_class_idxes=exclude_class_idxes;
        
end


