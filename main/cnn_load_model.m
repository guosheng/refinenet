

function [net_config, exp_info]=cnn_load_model(input_path)

  % input_path can be a cached model file, or a folder which contains the
  % model file and the file name follow the pattern: net-config*.mat

  net_config=[];
  exp_info=[];
      
  fprintf('check model: %s \n', input_path);
  if ~my_check_file(input_path)
    return;
  end
  
  if exist(input_path, 'dir')
      
      model_dir=input_path;
      net_config_file=fullfile(model_dir, 'net-config*.mat');
      net_config_file=my_findfile_by_pattern(net_config_file);

      info_file=fullfile(model_dir, 'exp-info*.mat');
      info_file=my_findfile_by_pattern(info_file);
           
      
  else
      
      net_config_file=input_path;
      info_file=[];
      
  end
  
  
  
  
  if isempty(net_config_file)
      return;
  end
  net_config=load(net_config_file);
  net_config=net_config.net_config;
  process_dagnn_load(net_config);
  fprintf('model loaded from: %s\n', net_config_file);
       
  
  if ~isempty(info_file)
    exp_info=load(info_file) ;
    exp_info=exp_info.exp_info;
    fprintf('exp_info loaded from: %s\n', info_file);
  end
 

end


function dag_net=fix_dagnn_name(dag_net)

check_file_names={'inputs'; 'outputs'; 'params'};

for l_idx=1:length(dag_net.layers)
    for f_idx=1:length(check_file_names)
        one_field_name=check_file_names{f_idx};
        one_v=dag_net.layers(l_idx).(one_field_name);
        if size(one_v, 1)>1
            dag_net.layers(l_idx).(one_field_name)=one_v';
        end
    end
end

end


function group_info=fix_layer_name(group_info)
  
  % for legacy model

    if ~isempty(group_info.net_info)
      net_info=group_info.net_info;
        for l_idx=1:length(net_info.ref.layers)
            tmp_layer=net_info.ref.layers{l_idx};
            if isfield(tmp_layer, 'custom_type')
              if strcmp(tmp_layer.custom_type, 'unary_softmaxloss')
                  tmp_layer.custom_type='dense_softmaxloss';
                  net_info.ref.layers{l_idx}=tmp_layer;
              end
            end
        end
    end
end


function process_dagnn_load(net_config)

assert(isa(net_config, 'ref_obj'));
  
for group_idx=1:length(net_config.ref.group_infos)
    group_info=net_config.ref.group_infos{group_idx};

    group_info=fix_layer_name(group_info);

    if check_group_dag_net(group_info)
        fprintf('processing dag_net for loading, group_idx:%d, group_name: %s\n', group_info.group_idx, group_info.name);
        group_info.dag_net=fix_dagnn_name(group_info.dag_net);
        group_info.dag_net = dagnn.DagNN.loadobj(group_info.dag_net) ;
        net_config.ref.group_infos{group_idx}=group_info;
    end
end

end


function use_dagnn=check_group_dag_net(group_info)
    
    use_dagnn=isfield(group_info, 'dag_net') && ~isempty(group_info.dag_net);
    if isfield(group_info, 'use_dagnn') && group_info.use_dagnn
        assert(use_dagnn==group_info.use_dagnn);
    end
    
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

