
function output_info=move_output_info_cpu(output_info)



if output_info.is_group_data
    
    data_child_groups=output_info.data_child_groups;
    output_valid_flags=output_info.child_valid_flags;
    data_child_groups=data_child_groups(output_valid_flags);
    child_num=length(data_child_groups);
    
    for g_idx=1:child_num
        data_child_groups{g_idx}=move_output_info_cpu(data_child_groups{g_idx});
    end
    
    output_info.data_child_groups(output_valid_flags)=data_child_groups;
    
else


    if isfield(output_info, 'x')
        output_info.x=move_to_cpu_values(output_info.x);
    end
    if isfield(output_info, 'aux')
        output_info.aux=move_to_cpu_values(output_info.aux);
    end
    if isfield(output_info, 'dzdx')
        output_info.dzdx=move_to_cpu_values(output_info.dzdx);
    end
    if isfield(output_info, 'dzdw')
        output_info.dzdw=move_to_cpu_values(output_info.dzdw);
    end
    
end
    
end
