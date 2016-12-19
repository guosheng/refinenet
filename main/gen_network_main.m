

function net_config=gen_network_main(train_opts)


fprintf('\n generate network...\n\n');
    
net_config=gen_init_net_config(train_opts);

net_config.ref.group_counter=0;

net_config.ref.root_group_info=My_net_util.gen_group_info_basic(net_config, 'root');
net_config.ref.root_group_info.independent_group=true;

group_output_info=gen_multipath_input_group(train_opts, net_config);
group_output_info=gen_network_refine_group(train_opts, net_config, group_output_info);
loss_group_info=gen_network_loss_group(train_opts, net_config, group_output_info);

net_config.ref.mc_prediction_group_idxes=loss_group_info.group_idx;
net_config.ref.valid_group_flags=true(length(net_config.ref.group_infos), 1);
net_config.ref.root_group_idx=net_config.ref.root_group_info.group_idx;

net_config.ref.group_infos{net_config.ref.root_group_info.group_idx, 1}=net_config.ref.root_group_info;

gen_dagnn_info(net_config);

    
end



function net_config=gen_init_net_config(train_opts)

net_config=[];
net_config.group_counter=0;
net_config.group_infos=cell(0);
net_config.root_group_idx=[];
net_config.valid_group_flags=[];

net_config.global_init_fn=@cnn_global_init_fn;

net_config.epoch_init_fn=[];
net_config.epoch_finish_fn=[];
net_config.batch_init_fn=[];
net_config.batch_finish_fn=[];

net_config.batch_info_disp_fn=@cnn_batch_info_disp;
net_config.epoch_info_disp_fn=[];

net_config.batch_evaluate_fn=[];
net_config.epoch_evaluate_fn=@cnn_epoch_evaluate;

net_config.epoch_run_config_fn=@my_epoch_run_config;
net_config.batch_run_config_fn=@my_batch_run_config;

net_config.net_progress_disp_fn=@my_net_progress_disp;

net_config.bp_start_epoch=1;

net_config.net_run_verbose=train_opts.net_run_verbose;


make_ref_obj(net_config);


end



function group_output_info=gen_multipath_input_group(train_opts, net_config)


inputpath_num=length(train_opts.inputpath_configs);

child_output_infos=cell(inputpath_num, 1);

multipath_input_group_info=My_net_util.gen_group_info_basic(net_config, 'multipath_input');
multipath_input_group_info.child_relation='parallel';


for path_idx=1:inputpath_num
           
    inputpath_config=train_opts.inputpath_configs{path_idx};
    [inputpath_group_info, one_net_output_info]=gen_inputpath_group_info(train_opts, net_config, inputpath_config);
    multipath_input_group_info.child_group_idxes(end+1,1)=inputpath_group_info.group_idx;
    child_output_infos{path_idx}=one_net_output_info;
end

net_config.ref.group_infos{multipath_input_group_info.group_idx,1}=multipath_input_group_info;
net_config.ref.root_group_info.child_group_idxes(end+1,1)=multipath_input_group_info.group_idx;

group_output_info=[];
group_output_info.child_infos=child_output_infos;

end






function gen_dagnn_info(net_config)


group_num=length(net_config.ref.group_infos);
dag_group_flags=false(group_num,1);


for g_idx=1:group_num

    one_group_info=net_config.ref.group_infos{g_idx};
    
    if isfield(one_group_info, 'use_dagnn') && one_group_info.use_dagnn
        dag_group_flags(g_idx)=true;
    end
end

net_config.ref.dag_group_flags=dag_group_flags;

   

end



function [inputpath_group_info, net_output_info]=gen_inputpath_group_info(train_opts, net_config, inputpath_config)
          
    dag_net=[];
    path_type=inputpath_config.path_type;
    
    net_output_info=[];

    if strcmp(path_type, 'resnet')
        [layers, net_output_info, dag_net]=gen_network_inputpath_resnet(inputpath_config);
    end

    if strcmp(path_type, 'vgg')
        error('not support!')
    end

    if strcmp(path_type, 'imgraw')
        [layers, net_output_info, dag_net]=gen_network_inputpath_imgraw(inputpath_config);
    end
        
    
    assert(~isempty(net_output_info));
    
    net_info=My_net_util.gen_net_info_basic(train_opts);
    net_info.ref.name=sprintf('inputpath_%d', inputpath_config.path_idx);
    net_info.ref.layers=layers;
    net_info.ref.lr_steps=net_info.ref.lr_steps.*inputpath_config.lr_factor;
    
        
    % control bp here:
    net_info.ref.do_bp=train_opts.input_net_do_bp;

    % or control the layer for bp:
    % net_info.ref.bp_start_layer=length(layers);

    % if init from a trained model, init will be performed up to this group
%     net_info.ref.stop_network_init=true;
        
    inputpath_group_info=My_net_util.gen_group_info_basic(net_config, net_info.ref.name);
    inputpath_group_info.net_info=net_info;
    inputpath_group_info.dag_net=dag_net;
    if ~isempty(dag_net)
        inputpath_group_info.use_dagnn=true;
    end
    net_config.ref.group_infos{inputpath_group_info.group_idx, 1}=inputpath_group_info;
    
end




