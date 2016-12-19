


function eva_result_epoch=update_eva_result_batch(eva_result_epoch, eva_result_batch)

    if isempty(eva_result_epoch)
        eva_result_epoch=eva_result_batch;
    else
        
        eva_names=eva_result_batch.eva_names;
        eva_names_disp=eva_result_batch.eva_names_disp;
                
        for eva_idx=1:length(eva_names)
            
            one_eva_name=eva_names{eva_idx};
            one_eva_name_disp=eva_names_disp{eva_idx};
            one_eva_sum_name=[one_eva_name '_sum'];
            one_eva_num_name=[one_eva_name '_eva_num'];
            
            if ~isfield(eva_result_epoch, one_eva_name)
                tmp_eva_idx=length(eva_result_epoch.eva_names)+1;
                eva_result_epoch.eva_names{tmp_eva_idx,1}=one_eva_name;
                eva_result_epoch.eva_names_disp{tmp_eva_idx,1}=one_eva_name_disp;
                eva_result_epoch.(one_eva_sum_name)=eva_result_batch.(one_eva_sum_name);
                eva_result_epoch.(one_eva_num_name)=eva_result_batch.(one_eva_num_name);
            else
                eva_result_epoch.(one_eva_sum_name)=...
                    eva_result_epoch.(one_eva_sum_name)+eva_result_batch.(one_eva_sum_name);
                eva_result_epoch.(one_eva_num_name)=...
                    eva_result_epoch.(one_eva_num_name)+eva_result_batch.(one_eva_num_name);
            end
            
            eva_result_epoch.(one_eva_name)=...
                eva_result_epoch.(one_eva_sum_name)/(eps + eva_result_epoch.(one_eva_num_name));
        end
        
    end

end




