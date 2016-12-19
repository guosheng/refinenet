




function input_info=cnn_layer_dense_softmaxloss_backward(input_info, layer, work_info_batch, output_info)

    stageout_info=layer.stageout_info;
    stage_idx=stageout_info.stage_idx;
    
    mc_info=work_info_batch.ref.finalloss_mc_info_stages{stage_idx};
            
    tmp_input_info=work_info_batch.ref.finalloss_tmp_input_info_stages{stage_idx};
    tmp_input_info=do_backward(tmp_input_info, output_info, mc_info);
    
    input_info.dzdx=tmp_input_info.dzdx;
                   
   
    example_non_valid_flags=mc_info.example_non_valid_flags;
   
      
    if ~isempty(example_non_valid_flags)
       
        valid_r_num=numel(example_non_valid_flags)-nnz(example_non_valid_flags);
                        
    else
        valid_r_num=mc_info.node_num;
    end

    
    if valid_r_num>0
        input_info.dzdx=input_info.dzdx./valid_r_num;
    end
        

end




function input_info=do_backward(input_info, output_info, mc_info)

c=mc_info.gt_label_data;

X=input_info.x;
dzdy=output_info.dzdx;

Y = vl_nnloss(X, c, dzdy, 'loss', 'softmaxlog') ;

input_info.dzdx=Y;
input_info.dzdw=[];

end



