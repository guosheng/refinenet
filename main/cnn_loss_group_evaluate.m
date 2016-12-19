

function cnn_loss_group_evaluate(work_info_batch, group_idx)
        
%     prediction_info=work_info_batch.ref.prediction_info_groups{group_idx};
%     obj_value=gather(prediction_info.output_infos{end}.x);
%     obj_value=sum(obj_value(:));
%     work_info_batch.ref.obj_value=obj_value;

    cnn_forward_evaluate_obj(work_info_batch, group_idx);
    cnn_softmax_evaluate(work_info_batch, group_idx);
    
    if ~work_info_batch.ref.run_eva
        return;
    end
       
    
    predict_info=gen_predict_info(work_info_batch, group_idx);
    do_full_eva(work_info_batch, predict_info);
    
end




function predict_info=gen_predict_info(work_info_batch, group_idx)
                
    prediction_info=work_info_batch.ref.prediction_info_groups{group_idx};
    one_output_info=prediction_info.output_infos{end};
    assert(check_valid_net_output(one_output_info));
    predict_info=one_output_info.mc_predict_info;
    
end


function do_full_eva(work_info_batch, predict_info)
    
    work_info_epoch=work_info_batch.ref.work_info_epoch;
    opts=work_info_batch.ref.train_opts;
        
    %assume no crop or data_aug
    batch_ds_info=work_info_batch.ref.ds_info;
    assert(isempty(batch_ds_info.aug_info));
    assert(isempty(batch_ds_info.crop_info));
        
    task_subidx=work_info_batch.ref.task_subidxes;
    img_idx=work_info_batch.ref.task_idxes;
    img_num=length(task_subidx);
    assert(img_num==1);
       
    
    seg_param=[];
    seg_param.eva_param=opts.eva_param;
    seg_param.img_idx=img_idx;
    
    batch_data=batch_ds_info.batch_data;
    seg_param.img_data_input=batch_data.img_data;
    seg_param.gt_mask_input=batch_data.label_data;
    seg_param.img_size_input=batch_data.img_size_origin;
    
    ds_info=work_info_batch.ref.imdb.ref.ds_info;
    eva_result_info=my_gen_seg_result(seg_param, ds_info, predict_info);
    
    work_info_epoch.ref.seg_eva_result_imgs_coarse{task_subidx}=eva_result_info.seg_eva_result_coarse;
    work_info_epoch.ref.seg_eva_result_imgs{task_subidx}=eva_result_info.seg_eva_result;
    work_info_epoch.ref.seg_eva_result_imgs_densecrf{task_subidx}=eva_result_info.seg_eva_result_densecrf;
       
    
end







