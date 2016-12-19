
function cnn_forward_evaluate_obj(work_info_batch, group_idx)


    prediction_info=work_info_batch.ref.prediction_info_groups{group_idx};
    one_output_info=prediction_info.output_infos{end};
    assert(check_valid_net_output(one_output_info));
    
    work_info_epoch=work_info_batch.ref.work_info_epoch;

    eva_result_epoch=work_info_epoch.ref.eva_result;
    eva_result_batch_obj=do_eva(one_output_info);
    eva_result_epoch=update_eva_result_batch(eva_result_epoch, eva_result_batch_obj);
    work_info_epoch.ref.eva_result=eva_result_epoch;
    
end





function eva_result_batch=do_eva( one_output_info)

    eva_result_batch=[];
       
    obj_values=sum(one_output_info.x(:));
    obj_values=gather(obj_values);
        
    my_check_valid_numeric(obj_values);
    
    obj_values=double(squeeze(obj_values));
       
    
    eva_result_batch.obj_value_sum=sum(obj_values(:));
    eva_result_batch.obj_value_eva_num=1;
    eva_result_batch.obj_value=eva_result_batch.obj_value_sum;
    
    eva_names=cell(0, 1);
    eva_names_disp=cell(0, 1);
    
    eva_names{end+1, 1}='obj_value';
    eva_names_disp{end+1, 1}='objective value';
    
    eva_result_batch.eva_names=eva_names;
    eva_result_batch.eva_names_disp=eva_names_disp;
    
end


