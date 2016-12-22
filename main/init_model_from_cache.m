

function init_model_from_cache(train_opts, net_config, net_config_cached)


train_opts.net_init_do_conv_layer_init_only=false;
train_opts.net_init_do_layer_check=true;

do_update_group_info(net_config, net_config_cached, train_opts);


end


function do_update_group_info(net_config, net_config_cached, train_opts)


assert(~isempty(net_config));
    
group_num=length(net_config.ref.group_infos);

for g_idx=1:group_num

    one_group_info=net_config.ref.group_infos{g_idx};
    
    [one_group_info, update_group_stop]=do_update_one_group(one_group_info, net_config_cached, train_opts);
    
    net_config.ref.group_infos{g_idx}=one_group_info;
    
    if update_group_stop
        break;
    end

end


end



function [one_group_info, update_group_stop]=do_update_one_group(one_group_info, net_config_cached, train_opts)


update_group_stop=false;

net_info=one_group_info.net_info;
if isempty(net_info)
    return;
end

if isfield(net_info.ref, 'stop_network_init') && net_info.ref.stop_network_init
    update_group_stop=true;
    return;
end


saved_group_info=net_config_cached.ref.group_infos{one_group_info.group_idx};
init_net_info=saved_group_info.net_info;

if isfield(net_info.ref, 'do_init_from_cache_fn')
    
    one_group_info=net_info.ref.do_init_from_cache_fn(one_group_info, net_info, init_net_info, train_opts);
    
else
    
    if train_opts.net_init_do_layer_check && ~isempty(net_info.ref.layers)
        try
            
            assert(length(net_info.ref.layers)==length(init_net_info.ref.layers));
        
            cached_layers=init_net_info.ref.layers;
            for l_idx=1:length(cached_layers)
                tmp_layer=net_info.ref.layers{l_idx};
                one_layer=cached_layers{l_idx};
                assert(strcmp(tmp_layer.type, one_layer.type));
                if isfield(tmp_layer, 'custom_type')
                    assert(strcmp(tmp_layer.custom_type, one_layer.custom_type));
                end
            end
        
        catch
            
            disp('the cached network layer is not matched!');
            keyboard;
        end
    end
    
    
    if train_opts.net_init_do_conv_layer_init_only
               

        cached_layers=init_net_info.ref.layers;
        for l_idx=1:length(cached_layers)
            one_layer=cached_layers{l_idx};
            if strcmp(one_layer.type, 'conv')
                assert(strcmp(net_info.ref.layers{l_idx}.type, 'conv'));
                net_info.ref.layers{l_idx}=one_layer;
            end
        end

    else
        
        net_info.ref.layers=init_net_info.ref.layers;
        
    end

end



one_group_info.net_info=net_info;


if isfield(one_group_info, 'use_dagnn') && one_group_info.use_dagnn
    one_group_info.dag_net=saved_group_info.dag_net;
end


if isfield(net_info.ref, 'init_from_cache_finished_fn')
    
    one_group_info=net_info.ref.init_from_cache_finished_fn(one_group_info, net_info, init_net_info, train_opts);
    
end
    

end




