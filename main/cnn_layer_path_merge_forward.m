

function output_info=cnn_layer_path_merge_forward(input_info, layer, work_info_batch)


assert(input_info.is_group_data);

input_info_child_groups=input_info.data_child_groups;
child_num=length(input_info_child_groups);

input_info_paths=cell(0);

for g_idx=1:child_num
    
    one_input_info=input_info_child_groups{g_idx};
    
    if one_input_info.is_group_data
        one_infos=one_input_info.data_child_groups;
    else
        one_infos={one_input_info};
    end
    
    input_info_paths=cat(1, input_info_paths, one_infos);
    
end

feat_sizes=cell(length(input_info_paths), 1);
for p_idx=1:length(input_info_paths)
    one_info=input_info_paths{p_idx};
    if isfield(one_info, 'x')
        one_size=size(one_info.x);
        feat_sizes{p_idx}=one_size;
    end
end

work_info_batch.ref.feat_map_size_inputpaths=feat_sizes;

output_info=[];
output_info.is_group_data=true;
output_info.data_child_groups=input_info_paths;

end



