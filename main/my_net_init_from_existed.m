

function my_net_init_from_existed(train_opts, net_config)


train_opts.net_init_do_conv_layer_init_only=false;
train_opts.net_init_do_layer_check=true;


if ~isempty(train_opts.net_init_model_path)
    
    fprintf('\n\n-------------------------------------------------------------\n\n');
    fprintf('net init from model:\n');
    disp(train_opts.net_init_model_path);
    
    do_init_net(train_opts, net_config);
    fprintf('\n\n-------------------------------------------------------------\n\n');
end


end




function net_config=do_init_net(train_opts, net_config)

    net_init_model_path=train_opts.net_init_model_path;
    
    saved_net_config=cnn_do_load_net(net_init_model_path);
    if isempty(saved_net_config)
        error('net-config init file is not existed!!!!');
    end
    
    if isfield(saved_net_config.ref, 'dag_group_flags')
      assert(all(net_config.ref.dag_group_flags==saved_net_config.ref.dag_group_flags));
    end
    
    do_update_group_info(net_config, saved_net_config, train_opts);
    
end





function do_update_group_info(net_config, saved_net_config, train_opts)


assert(~isempty(net_config));
    
group_num=length(net_config.ref.group_infos);

for g_idx=1:group_num

    one_group_info=net_config.ref.group_infos{g_idx};
    
    [one_group_info, update_group_stop]=do_update_one_group(one_group_info, saved_net_config, train_opts);
    
               
    net_config.ref.group_infos{g_idx}=one_group_info;
    
    if update_group_stop
        break;
    end

end




end






function [one_group_info, update_group_stop]=do_update_one_group(one_group_info, saved_net_config, train_opts)


update_group_stop=false;

net_info=one_group_info.net_info;
if isempty(net_info)
    return;
end

if isfield(net_info.ref, 'stop_network_init') && net_info.ref.stop_network_init
    update_group_stop=true;
    return;
end


saved_group_info=saved_net_config.ref.group_infos{one_group_info.group_idx};
init_net_info=saved_group_info.net_info;

if isfield(net_info.ref, 'net_info_init_fn')
    
    one_group_info=net_info.ref.net_info_init_fn(one_group_info, net_info, init_net_info, train_opts);
    
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


if isfield(net_info.ref, 'net_info_init_finished_fn')
    
    one_group_info=net_info.ref.net_info_init_finished_fn(one_group_info, net_info, init_net_info, train_opts);
    
end
    

end




