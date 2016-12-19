

function cnn_save_snapshot(epoch_dir, net_config, exp_info)


saved_net_config=[];
make_ref_obj(saved_net_config);
saved_net_config.ref.group_infos=net_config.ref.group_infos;
saved_net_config.ref.dag_group_flags=net_config.ref.dag_group_flags;
cnn_update_net_group(saved_net_config, @do_save_net_process, []);


saved_net_config=saved_net_config.ref;
saved_net_config.cache_filename=net_config.ref.cache_filename;
do_save_model(epoch_dir, saved_net_config, exp_info)


end





function do_save_model(epoch_dir, saved_net_config, exp_info)

    if ~isempty(saved_net_config)
                        
        dag_group_flags=saved_net_config.dag_group_flags;
        if any(dag_group_flags)
            dag_group_idxes=find(dag_group_flags);
            for d_idx=1:length(dag_group_idxes)
                group_idx=dag_group_idxes(d_idx);
                group_info=saved_net_config.group_infos{group_idx};
                
                cache_file=sprintf('group_%s_%d.mat', group_info.name, group_info.group_idx);
                cache_file=fullfile(epoch_dir, cache_file);
                
                dag_net=group_info.dag_net;
                device_org=dag_net.device;
                saved_net = dag_net.saveobj() ;
                

                group_info.dag_net=[];
                saved_net_config.group_infos{group_idx}=group_info;
                

                fprintf('save dag_net to: %s\n', cache_file);
                save(cache_file, '-struct', 'saved_net', '-v7.3') ;
                
                if strcmp(device_org, 'gpu')
                   dag_net.move('gpu');
                end
            end
        end
        
        
        net_filename=saved_net_config.cache_filename;
        saved_net_config.cache_filename=[];
        net_config_file=fullfile(epoch_dir, net_filename);
        fprintf('save net_config to: %s\n', net_config_file);
        try
            save(net_config_file, 'saved_net_config', '-v7.3') ;
        catch err_info
            disp(err_info.getReport);
            dbstack;
            keyboard;
        end
                
    end
        

    if ~isempty(exp_info)
        info_filename=exp_info.cache_filename;
        info_file=fullfile(epoch_dir, info_filename);
        fprintf('save exp_info to: %s\n', info_file);
        try
        	save(info_file, 'exp_info', '-v7.3') ;
        catch err_info
        	disp(err_info.getReport);
        	dbstack;
        	keyboard;
        end
    end

end








function one_group_info=do_save_net_process(input_group_info, opts)


net_info=input_group_info.net_info;
one_group_info=[];

if isempty(net_info)
    return;
end

new_net_info=[];
make_ref_obj(new_net_info);
new_net_info.ref.layers=net_info.ref.layers;


for i=1:numel(new_net_info.ref.layers)
    
    if strcmp(new_net_info.ref.layers{i}.type,'conv')
          new_net_info.ref.layers{i}.filtersMomentum = [];
          new_net_info.ref.layers{i}.biasesMomentum = [];
    end
    
    if strcmp(new_net_info.ref.layers{i}.type,'my_custom')
        layer_update_fn=new_net_info.ref.layers{i}.layer_update_fn;
        if ~isempty(layer_update_fn)
            update_info=[];
            update_info.group_info=input_group_info;
            new_net_info.ref.layers{i}=new_net_info.ref.layers{i}.layer_update_fn(...
                'cache_clean', new_net_info.ref.layers{i}, update_info);
        end
    end
end


if net_info.ref.net_on_gpu
    my_move_net(new_net_info, 'cpu') ;
end


one_group_info.net_info=new_net_info;
one_group_info.name=input_group_info.name;
one_group_info.group_idx=input_group_info.group_idx;


if isfield(input_group_info, 'use_dagnn') && input_group_info.use_dagnn
    one_group_info.dag_net=input_group_info.dag_net;
end


end

