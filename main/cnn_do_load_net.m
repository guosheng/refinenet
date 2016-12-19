
function saved_net_config=cnn_do_load_net(snapshot_dir)

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
%                 load(cache_file, 'net', 'stats') ;
%                 net = dagnn.DagNN.loadobj(net) ;
                net = dagnn.DagNN.loadobj(load(cache_file)) ;
                group_info.dag_net=net;
                saved_net_config.group_infos{group_idx}=group_info;
            end
        end
  end
  
  make_ref_obj(saved_net_config);
  
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



