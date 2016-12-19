


function cnn_update_net_group(net_config, update_fn, update_opts)


assert(~isempty(net_config));
    
group_num=length(net_config.ref.group_infos);

for g_idx=1:group_num

    one_group_info=net_config.ref.group_infos{g_idx};
    
    one_group_info=update_fn(one_group_info, update_opts);
    
    net_config.ref.group_infos{g_idx}=one_group_info;

end


end


