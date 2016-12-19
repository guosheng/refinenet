

function input_info = cnn_layer_path_merge_backward(input_info, layer, work_info_batch, output_info)



assert(input_info.is_group_data);
child_num=length(input_info.data_child_groups);

data_counter=0;

for g_idx=1:child_num
    
    one_input_info=input_info.data_child_groups{g_idx};
    
    if one_input_info.is_group_data
        one_data_num=length(one_input_info.data_child_groups);
        
        for inner_idx=1:one_data_num
            data_idx=data_counter+inner_idx;
            one_dzdx=output_info.data_child_groups{data_idx}.dzdx;
            one_input_info.data_child_groups{inner_idx}.dzdx=one_dzdx;
        end
        
    else
        one_data_num=1;
        data_idx=data_counter+one_data_num;
        one_dzdx=output_info.data_child_groups{data_idx}.dzdx;
        one_input_info.dzdx=one_dzdx;
    end
        
    data_counter=data_counter+one_data_num;
    
    input_info.data_child_groups{g_idx}=one_input_info;
    
end

input_info.bp_child_valid_flags=true(child_num,1);


end



