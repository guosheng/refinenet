
function cnn_softmax_evaluate(work_info_batch, group_idx)

    work_info_epoch=work_info_batch.ref.work_info_epoch;
    opts=work_info_batch.ref.train_opts;
    eva_result_batch_mc=do_evaluate_mc(opts, work_info_batch, group_idx);
    
    if ~isempty(eva_result_batch_mc)
        eva_result_epoch=work_info_epoch.ref.eva_result;
        eva_result_epoch=update_eva_result_batch(eva_result_epoch, eva_result_batch_mc);
        work_info_epoch.ref.eva_result=eva_result_epoch;
    end
    
end
   
    


function eva_result_batch=do_evaluate_mc(opts, work_info_batch, group_idx)
    
    eva_result_batch=[];
    
    
    prediction_info=work_info_batch.ref.prediction_info_groups{group_idx};
    one_output_info=prediction_info.output_infos{end};
    assert(check_valid_net_output(one_output_info));
    
   
    predict_info=one_output_info.mc_predict_info;
    mc_info=predict_info.mc_info;
    mc_scores=predict_info.score_map;
    
    e_num=size(mc_scores, 1)*size(mc_scores, 2);
    assert(size(mc_scores, 3)==mc_info.class_num);
    assert(e_num==mc_info.e_num);
    
    
    [~, predict_class_idxes]=max(mc_scores, [], 3);
    predict_class_idxes=gather(predict_class_idxes);
    
    gt_class_idxes=mc_info.gt_label_data;
    error_flags=gt_class_idxes~=predict_class_idxes;
    
    non_valid_flags=mc_info.example_non_valid_flags;
    if ~isempty(non_valid_flags)
        error_flags(non_valid_flags)=false;
    end
      
 
    error_mc_sum=nnz(error_flags);
    error_mc_eva_num=mc_info.valid_e_num+eps;
            
    eva_names=cell(0, 1);
    eva_names_disp=cell(0, 1);
        
    eva_result_batch.error_mc_sum=error_mc_sum;
    eva_result_batch.error_mc_eva_num=error_mc_eva_num;
    eva_result_batch.error_mc=error_mc_sum/error_mc_eva_num;
    
    eva_names{end+1, 1}='error_mc';
    eva_names_disp{end+1, 1}='multiclass error';
            
    eva_result_batch.eva_names=eva_names;
    eva_result_batch.eva_names_disp=eva_names_disp;
    
            
end





