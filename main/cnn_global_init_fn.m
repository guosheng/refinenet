
function cnn_global_init_fn(train_opts, imdb, work_info, net_config)

input_task_num=length(work_info.ref.task_info.task_idxes);
work_info.ref.input_task_num=input_task_num;

do_init_net_info(train_opts, work_info, net_config);
do_init_dagnn(train_opts, work_info, net_config)


end



function do_init_net_info(train_opts, work_info, net_config)

if ~work_info.ref.run_trn
    return;
end
    

group_num=length(net_config.ref.group_infos);
optimizer_param=train_opts.optimizer_param;

for g_idx=1:group_num
    group_info=net_config.ref.group_infos{g_idx};
    net_info=group_info.net_info;
    
    if ~isempty(net_info)
                
        assert(isa(net_info, 'ref_obj'));
        
        one_optimizer_param=optimizer_param;
        one_optimizer_param.learning_rate=net_info.ref.lr_multiplier * optimizer_param.learning_rate;
        
        net_info.ref.tmp_data.optimizer_param=one_optimizer_param;
    end
end

end




function do_init_dagnn(train_opts, work_info, net_config)


dag_group_flags=net_config.ref.dag_group_flags;
if ~any(dag_group_flags)
    return;
end

group_num=length(net_config.ref.group_infos);
state_groups=cell(group_num, 1);
run_trn=work_info.ref.run_trn;

dag_group_idxes=find(dag_group_flags);
use_gpu=train_opts.use_gpu;

for g_idx_idx=1:length(dag_group_idxes)

    g_idx=dag_group_idxes(g_idx_idx);
    one_group_info=net_config.ref.group_infos{g_idx};
    
    dag_net=one_group_info.dag_net;
    
    if use_gpu
          dag_net.move('gpu') ;
    end
        
    if run_trn
        state=[];
        state.momentum = num2cell(zeros(1, numel(dag_net.params))) ;
        
        if use_gpu
          state.momentum = cellfun(@gpuArray,state.momentum,'UniformOutput',false) ;
        end
        
        state.optimizer_param=one_group_info.net_info.ref.tmp_data.optimizer_param;
       
        state_groups{g_idx}=state;
    end
    
    config_batch_norm_resnet(dag_net, train_opts);
    
end

if run_trn
    work_info.ref.tmp_cache.dag_state_groups=state_groups;
end
  

end





function config_batch_norm_resnet(dag_net, train_opts)
    

    for l_idx=1:length(dag_net.layers)
        l=dag_net.layers(l_idx);
        block=l.block;

        if isa(block, 'dagnn.BatchNorm')
             if ~isempty(block.bnorm_moment_type_trn)                 
                 assert(strcmp(block.bnorm_moment_type_trn,train_opts.bnorm_moment_type_trn));
             end
             if ~isempty(block.bnorm_moment_type_tst)                 
                 assert(strcmp(block.bnorm_moment_type_tst,train_opts.bnorm_moment_type_tst));
             end
             block.bnorm_moment_type_trn=train_opts.bnorm_moment_type_trn;
             block.bnorm_moment_type_tst=train_opts.bnorm_moment_type_tst;

             dag_net.layers(l_idx).block=block;

        end
    end
   

end





