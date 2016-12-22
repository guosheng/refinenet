

function cnn_save_model(epoch_dir, net_config_input, exp_info)


if ~isempty(net_config_input)
    net_config=gen_saved_net_config(net_config_input);
    net_filename=net_config.ref.cache_filename;
    net_config.ref.cache_filename=[];
    net_config_file=fullfile(epoch_dir, net_filename);
    fprintf('save net_config to: %s\n', net_config_file);
    try
        save(net_config_file, 'net_config', '-v7.3') ;
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



function net_config=gen_saved_net_config(net_config_input)


assert(isa(net_config_input, 'ref_obj'));

net_config=[];
make_ref_obj(net_config);
net_config.ref=net_config_input.ref;
  
for group_idx=1:length(net_config.ref.group_infos)
    group_info=net_config.ref.group_infos{group_idx};
    
    net_info=group_info.net_info;
    if ~isempty(net_info)
        new_net_info=[];
        make_ref_obj(new_net_info);
        new_net_info.ref=net_info.ref;
        do_clean_net_info(new_net_info);
        if new_net_info.ref.net_on_gpu
            my_move_net(new_net_info, 'cpu') ;
        end
        group_info.net_info=new_net_info;
    end
      
    
    if check_group_dag_net(group_info)
        
        fprintf('processing dag_net for saving, group_idx:%d, group_name: %s\n', group_info.group_idx, group_info.name);
        
        dag_net=group_info.dag_net;
        device_org=dag_net.device;
        saved_dag_net = dag_net.saveobj() ;

        if strcmp(device_org, 'gpu')
           dag_net.move('gpu');
        end
        
        group_info.dag_net=saved_dag_net;
    end
    
    
    net_config.ref.group_infos{group_idx}=group_info;
end


end




function do_clean_net_info(new_net_info)

new_net_info.ref.tmp_data=[];

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


end

function use_dagnn=check_group_dag_net(group_info)
    
    use_dagnn=isfield(group_info, 'dag_net') && ~isempty(group_info.dag_net);
    if isfield(group_info, 'use_dagnn') && group_info.use_dagnn
        assert(use_dagnn==group_info.use_dagnn);
    end
    
end




