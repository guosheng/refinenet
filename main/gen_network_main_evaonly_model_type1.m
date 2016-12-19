

function net_config=gen_network_main_evaonly(train_opts)

% this file is modified from gen_network_main.m
% if any configurations change in gen_network_main.m, this file should be modified accordingly

fprintf('\n generate network evaonly...\n\n');
    
net_config=gen_init_net_config(train_opts);

net_config.ref.group_counter=0;

net_config.ref.root_group_info=My_net_util.gen_group_info_basic(net_config, 'root');
net_config.ref.root_group_info.independent_group=true;

gen_multipath_input_group_evaonly(train_opts, net_config);
gen_network_refine_group_evaonly(train_opts, net_config);
loss_group_info=gen_network_loss_group_evaonly(train_opts, net_config);

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



function group_output_info=gen_multipath_input_group_evaonly(train_opts, net_config)


inputpath_num=length(train_opts.inputpath_configs);

child_output_infos=cell(inputpath_num, 1);

multipath_input_group_info=My_net_util.gen_group_info_basic(net_config, 'multipath_input');
multipath_input_group_info.child_relation='parallel';


for path_idx=1:inputpath_num
           
    inputpath_config=train_opts.inputpath_configs{path_idx};
    inputpath_group_info=gen_inputpath_group_info_evaonly(train_opts, net_config, inputpath_config);
    multipath_input_group_info.child_group_idxes(end+1,1)=inputpath_group_info.group_idx;
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



function inputpath_group_info=gen_inputpath_group_info_evaonly(train_opts, net_config, inputpath_config)
        
    net_info=My_net_util.gen_net_info_basic(train_opts);
    net_info.ref.name=sprintf('inputpath_%d', inputpath_config.path_idx);
    net_info.ref.layers=[];
        
    inputpath_group_info=My_net_util.gen_group_info_basic(net_config, net_info.ref.name);
    inputpath_group_info.net_info=net_info;
    inputpath_group_info.dag_net=[];
    inputpath_group_info.use_dagnn=true;
    
    net_config.ref.group_infos{inputpath_group_info.group_idx, 1}=inputpath_group_info;
    
    
    
end







function gen_network_refine_group_evaonly(train_opts, net_config)

multipath_fusion_group_info=My_net_util.gen_group_info_basic(net_config, 'multipath_fusion');

net_info=My_net_util.gen_net_info_basic(train_opts);
net_info.ref.name='multipath_fusion';
net_info.ref.layers=[];

multipath_fusion_group_info.net_info=net_info;
multipath_fusion_group_info.use_dagnn=true;

net_config.ref.group_infos{multipath_fusion_group_info.group_idx,1}=multipath_fusion_group_info;
net_config.ref.root_group_info.child_group_idxes(end+1,1)=multipath_fusion_group_info.group_idx;


end




function loss_group_info=gen_network_loss_group_evaonly(train_opts, net_config)



loss_group_info=My_net_util.gen_group_info_basic(net_config, 'loss_group');
loss_group_info.forward_evaluate_fn=@cnn_loss_group_evaluate;

net_info=My_net_util.gen_net_info_basic(train_opts);
net_info.ref.name='final_loss';
net_info.ref.layers=[];

net_info.ref.net_info_init_finished_fn=@loss_group_init_finished_fn;

loss_group_info.net_info=net_info;
loss_group_info.use_dagnn=false;

if train_opts.loss_config.lossgroup_conv_num>0
    loss_group_info.use_dagnn=true;
end

loss_group_info.prediction_layer_idxes=[];

net_config.ref.group_infos{loss_group_info.group_idx,1}=loss_group_info;
net_config.ref.root_group_info.child_group_idxes(end+1,1)=loss_group_info.group_idx;

end


function one_group_info=loss_group_init_finished_fn(one_group_info, net_info, init_net_info, train_opts)

one_group_info.prediction_layer_idxes=length(net_info.ref.layers);


end


