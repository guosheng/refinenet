


function my_load_legacy_model(snapshot_dir, net_config)

      
  fprintf('check snapshot: %s \n', snapshot_dir);
    
  exp_info=[];
  saved_net_config=do_load_net(snapshot_dir);

  if isempty(saved_net_config)
      return;
  end
  
  if isfield(saved_net_config.ref, 'dag_group_flags')
      assert(all(net_config.ref.dag_group_flags==saved_net_config.ref.dag_group_flags));
  end
  cnn_update_net_group(net_config, @do_load_net_process_snapshot, saved_net_config);
  
   
  info_file=fullfile(snapshot_dir, 'exp-info*.mat');
  info_file=my_findfile_by_pattern(info_file);
  if ~isempty(info_file)
    tmp_load_data=my_load_file(info_file) ;
    exp_info=tmp_load_data.exp_info;
  end
    
  net_config.ref.init_exp_info=exp_info;
  
  fprintf('loaded snpashot: %s\n', snapshot_dir);
  

end



function one_group_info=do_load_net_process_snapshot(one_group_info, saved_net_config)



net_info=one_group_info.net_info;
if isempty(net_info)
    return;
end


saved_group_info=saved_net_config.ref.group_infos{one_group_info.group_idx};

if ~isempty(net_info.ref.layers)
    net_info.ref.layers=saved_group_info.net_info.ref.layers;
end

if isfield(one_group_info, 'use_dagnn') && one_group_info.use_dagnn
    one_group_info.dag_net=saved_group_info.dag_net;
end


one_group_info.net_info=net_info;


end






function result=my_findfile_by_pattern(net_config_file)

  result=[];
  
  file_info=dir(net_config_file);
  if isempty(file_info)
      return
  end
  file_info=file_info(1);
  onedir=fileparts(net_config_file);
  result=fullfile(onedir, file_info.name);
    
end





function saved_net_config=do_load_net(snapshot_dir)

    saved_net_config=[];

    if ~my_check_file(snapshot_dir)
      return;
    end
  
  net_config_file=fullfile(snapshot_dir, 'net-config*.mat');
  net_config_file=my_findfile_by_pattern(net_config_file);
  if isempty(net_config_file)
      return;
  end
    
  
  tmp_load_data=my_load_file(net_config_file) ;
  saved_net_config=tmp_load_data.saved_net_config;
  
  if isfield(saved_net_config, 'dag_group_flags')
        dag_group_flags=saved_net_config.dag_group_flags;
        if any(dag_group_flags)
            dag_group_idxes=find(dag_group_flags);
            for d_idx=1:length(dag_group_idxes)
                group_idx=dag_group_idxes(d_idx);
                group_info=saved_net_config.group_infos{group_idx};
                cache_file=sprintf('group_%s_%d.mat', group_info.name, group_info.group_idx);
                cache_file=fullfile(snapshot_dir, cache_file);

                fprintf('load dag_net:: %s\n', cache_file);
                net = dagnn.DagNN.loadobj(load(cache_file)) ;
                group_info.dag_net=net;
                saved_net_config.group_infos{group_idx}=group_info;
            end
        end
  end
  
  make_ref_obj(saved_net_config);
  
end







