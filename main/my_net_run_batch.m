



function my_net_run_batch(net_config, work_info_batch)
        
    gen_data_info_groups_cache(net_config, work_info_batch);
    group_idx=net_config.ref.root_group_idx;

    data_info=init_one_data_info_from_cache(group_idx, work_info_batch);
    input_info=work_info_batch.ref.ds_info.net_input_info;
    data_info.ref.output_info_layers{1}=input_info;
    work_info_batch.ref.ds_info.net_input_info=[];
        
    do_run_one_pass(net_config, work_info_batch, group_idx);        
    
    work_info_batch.ref.data_info_groups=[];
            
end
    


function do_run_one_pass(net_config, work_info_batch, group_idx)

    if work_info_batch.ref.net_run_verbose
        fprintf('---- runing one pass, root_group_idx:%d\n', group_idx);
    end
    
    do_run_one_group(net_config, work_info_batch, group_idx, 'forward');
        
    data_info=work_info_batch.ref.data_info_groups{group_idx};
    last_layer_output_info=data_info.ref.output_info_layers{end};
    assert(check_valid_net_output(last_layer_output_info));

    if work_info_batch.ref.run_backward

        init_dzdx=single(1);

        last_layer_output_info.dzdx=init_dzdx;
        last_layer_output_info.bp_finished=true;
        data_info.ref.output_info_layers{end}=last_layer_output_info;

        do_run_one_group(net_config, work_info_batch, group_idx, 'backward');
    end
 
end

    

function do_run_one_group(net_config, work_info_batch, group_idx, group_run_type)


group_info=net_config.ref.group_infos{group_idx};
work_info_batch.ref.net_run_current_group_idx=group_idx;

net_run_current_group_idx_linked=[];
if ~isempty(group_info.net_info)
    if isfield(group_info.net_info.ref, 'linked_group_idx') 
        net_run_current_group_idx_linked=group_info.net_info.ref.linked_group_idx;
    end
end
work_info_batch.ref.net_run_current_group_idx_linked=net_run_current_group_idx_linked;



if work_info_batch.ref.net_run_verbose
    fprintf('----start group_idx:%d, name:%s, run_type:%s\n', ...
        group_info.group_idx, group_info.name, group_run_type);
end


if strcmp(group_run_type, 'forward')
    do_forward_one_group( net_config, work_info_batch, group_idx);
end

if strcmp(group_run_type, 'backward')
    do_backward_one_group( net_config, work_info_batch, group_idx);
end


if work_info_batch.ref.net_run_verbose
    fprintf('----finish group_idx:%d, name:%s, run_type:%s\n', ...
        group_info.group_idx, group_info.name, group_run_type);
end
    

end







function do_forward_group_child_chain( work_info_batch, data_info, group_info)

        
        net_config=work_info_batch.ref.net_config;

        child_group_idxes=group_info.child_group_idxes;

        child_num=length(child_group_idxes);
        child_valid_flags=data_info.ref.child_valid_flags;
        assert(any(child_valid_flags));
        
        input_info=data_info.ref.output_info_layers{1};
        assert(check_valid_net_output(input_info));
        need_bp=false;
       
        for g_idx=1:child_num
            
            child_valid=child_valid_flags(g_idx);
            if ~child_valid
                break;
            end
            
            child_group_idx=child_group_idxes(g_idx);
            
            [input_info, need_bp]=do_forward_one_child( net_config, ...
                work_info_batch, child_group_idx, input_info);
            
            if ~check_valid_net_output(input_info)
                break;
            end
        end
                
        data_info.ref.output_info_layers{2}=input_info;
        data_info.ref.need_bp=need_bp;
        
end






function do_forward_group_child_parallel( work_info_batch, data_info, group_info)
   
    net_config=work_info_batch.ref.net_config;

    child_group_idxes=group_info.child_group_idxes;
    child_num=length(child_group_idxes);
    child_valid_flags=data_info.ref.child_valid_flags;
    assert(any(child_valid_flags));

    group_input_info=data_info.ref.output_info_layers{1};

    assert(check_valid_net_output(group_input_info));

    assert(group_input_info.is_group_data);
    
        
    
    
    input_info_child_groups=group_input_info.data_child_groups;
    input_child_valid_flags=group_input_info.child_valid_flags;
    assert(length(input_info_child_groups)==child_num);
    assert(length(input_child_valid_flags)==child_num);
    
   

    output_child_groups=cell(child_num, 1);
    output_valid_flags=false(child_num,1);
    need_bp_flags=false(child_num,1);
    
    for g_idx=1:child_num

        child_valid=child_valid_flags(g_idx);
        if ~child_valid
            continue;
        end
                
        input_valid=input_child_valid_flags(g_idx);
        if ~input_valid
            continue;
        end
        child_input_info=input_info_child_groups{g_idx};
        
        child_group_idx=child_group_idxes(g_idx);
        
        [child_output_info, child_need_bp]=do_forward_one_child(net_config, ...
            work_info_batch, child_group_idx, child_input_info);
        output_child_groups{g_idx}=child_output_info;
        
        need_bp_flags(g_idx)=child_need_bp;

        output_valid_flags(g_idx)=check_valid_net_output(child_output_info);
    end

    assert(any(output_valid_flags));

    group_output_info=[];
    group_output_info.data_child_groups=output_child_groups;
    group_output_info.child_valid_flags=output_valid_flags;
    group_output_info.is_group_data=true;
    group_output_info.forward_finished=any(output_valid_flags);


    data_info.ref.output_info_layers{2}=group_output_info;
    data_info.ref.need_bp=any(need_bp_flags);
    data_info.ref.need_bp_child_groups=need_bp_flags;
    
end











function do_forward_one_group( net_config, work_info_batch, group_idx)



group_info=net_config.ref.group_infos{group_idx};
prediction_layer_idxes=group_info.prediction_layer_idxes;


if ~isempty(group_info.forward_begin_fn)
    group_info=group_info.forward_begin_fn(work_info_batch, group_info);
end

if ~group_info.skip_forward
            
    
    child_group_idxes=group_info.child_group_idxes;
    data_info=work_info_batch.ref.data_info_groups{group_idx};
     
      

    if isempty(child_group_idxes)
                        
        net_run_config=work_info_batch.ref.net_run_config;
        
        
        extra_output_layer_idxes=[];
        if ~isempty(prediction_layer_idxes)
            extra_output_layer_idxes=cat(1, extra_output_layer_idxes, prediction_layer_idxes);
            extra_output_layer_idxes=unique(extra_output_layer_idxes);
        end
        
        
        work_info_batch.ref.current_group_data_info=data_info;
                                
        forward_time=tic;
        extra_data_info=my_net_forward(retrieve_net_info(group_info, net_config), ...
            work_info_batch, data_info, net_run_config, extra_output_layer_idxes);
        forward_time=toc(forward_time);
        work_info_batch.ref.forward_time=work_info_batch.ref.forward_time+forward_time;    
        
        data_info.ref.extra_input_infos=[];
        
        work_info_batch.ref.current_group_data_info=[];
        
    else


        switch group_info.child_relation
            case 'chain'
                do_forward_group_child_chain( work_info_batch, data_info, group_info);
            case 'parallel'
                do_forward_group_child_parallel( work_info_batch, data_info, group_info);
            otherwise
                error('not support!');
        end

    end
end



if ~isempty(prediction_layer_idxes)
    
    assert(isempty(child_group_idxes));
    
    prediction_info=work_info_batch.ref.prediction_info_groups{group_idx};
    keep_output_infos=extra_data_info.output_layers(group_info.prediction_layer_idxes);

    prediction_info.output_infos=keep_output_infos;
    work_info_batch.ref.prediction_info_groups{group_idx}=prediction_info;
end



if ~isempty(group_info.forward_finish_fn)
    group_info=group_info.forward_finish_fn(work_info_batch, group_info);
end

if ~isempty(group_info.forward_evaluate_fn)
    forward_eva_time=tic;
    group_info.forward_evaluate_fn( work_info_batch, group_idx);
    forward_eva_time=toc(forward_eva_time);
    work_info_batch.ref.eva_time=work_info_batch.ref.eva_time+forward_eva_time;    
end



end






function [output_info, child_need_bp]=do_forward_one_child(net_config,...
    work_info_batch, child_group_idx, input_info)

    child_data_info=init_one_data_info_from_cache(child_group_idx, work_info_batch);
    child_data_info.ref.output_info_layers{1}=input_info;

    do_run_one_group(net_config, work_info_batch, child_group_idx, 'forward');
    output_info=child_data_info.ref.output_info_layers{end};
        
    child_data_info.ref.output_info_layers{end}=[];
    child_need_bp=child_data_info.ref.need_bp;
        
    if ~child_need_bp
        child_data_info.ref.output_info_layers=[];
    end
            

end



function output_info_layers=gen_net_output_info_basic(net_info)

if isempty(net_info)
    n=1;
else
    n = numel(net_info.ref.layers) ;
end


sample_output_info=[];
sample_output_info.x=[];
sample_output_info.aux=[];
sample_output_info.dzdx=[];
sample_output_info.dzdw=[];
sample_output_info.is_group_data=false;
sample_output_info.bp_finished=false;
sample_output_info.forward_finished=false;

output_info_layers=repmat({sample_output_info}, n+1, 1);


end






function data_info=init_one_data_info_from_cache(group_idx, work_info_batch)

data_info=work_info_batch.ref.data_info_groups{group_idx};
data_info.ref=work_info_batch.ref.sample_data_info_groups{group_idx};

tmp_data_info=work_info_batch.ref.data_info_groups_extra{group_idx};
data_info.ref.extra_input_infos=tmp_data_info.extra_input_infos;

end



function data_info=init_one_data_info(group_info, net_config)

    data_info=[];
    data_info.output_info_layers=[];
    data_info.forward_time=0;
    data_info.backward_time=0;
    data_info.need_bp=false;
    data_info.extra_input_infos=cell(0);
    
    child_num=length(group_info.child_group_idxes);
    data_info.child_valid_flags=true(child_num, 1);
        
    data_info.output_info_layers=gen_net_output_info_basic(...
        retrieve_net_info(group_info, net_config));
    
    

end



function gen_data_info_groups_cache(net_config, work_info_batch)


    group_num=length(net_config.ref.group_infos);
    work_info=work_info_batch.ref.work_info;
    if ~isfield(work_info.ref.tmp_cache, 'sample_data_info_groups')
        data_info_groups=cell(group_num, 1);
        data_info_groups_ref=cell(group_num, 1);
        for g_idx=1:group_num
            tmp_data_info=[];
            make_ref_obj(tmp_data_info);
            data_info_groups_ref{g_idx}=tmp_data_info;
            data_info=init_one_data_info(net_config.ref.group_infos{g_idx}, net_config);
            data_info_groups{g_idx}=data_info;
        end
        work_info.ref.tmp_cache.sample_data_info_groups=data_info_groups;
        work_info.ref.tmp_cache.sample_data_info_groups_ref=data_info_groups_ref;
        
        
        data_info_groups_extra=cell(group_num, 1);
        for g_idx=1:group_num
            tmp_data_info_extra=[];
            tmp_data_info_extra.extra_input_infos=cell(0);
            data_info_groups_extra{g_idx}=tmp_data_info_extra;
        end
        work_info.ref.tmp_cache.sample_data_info_groups_extra=data_info_groups_extra;
        
    end
    
    
    
    work_info_batch.ref.sample_data_info_groups=work_info.ref.tmp_cache.sample_data_info_groups;
    work_info_batch.ref.data_info_groups=work_info.ref.tmp_cache.sample_data_info_groups_ref;
    work_info_batch.ref.prediction_info_groups=cell(group_num, 1);
    
    work_info_batch.ref.data_info_groups_extra=work_info.ref.tmp_cache.sample_data_info_groups_extra;
    
end








function do_backward_group_child_chain(net_config,work_info_batch, data_info, group_info)

    child_group_idxes=group_info.child_group_idxes;
    child_num=length(child_group_idxes);
    child_valid_flags=data_info.ref.child_valid_flags;
    assert(any(child_valid_flags));

      
        start_g_idx=find(~child_valid_flags, 1);
        if isempty(start_g_idx)
            start_g_idx=child_num;
        end

        init_output_info=data_info.ref.output_info_layers{end};
        
        for g_idx=start_g_idx:-1:1
                        
            child_group_idx=child_group_idxes(g_idx);
            init_output_info=do_backward_child_group( net_config,...
                work_info_batch, child_group_idx, init_output_info);
           
            if ~init_output_info.bp_finished
                break;
            end
        end
        
        data_info.ref.output_info_layers{1}=init_output_info;
        
end




function do_backward_group_child_parallel(net_config,work_info_batch,data_info, group_info)

    child_group_idxes=group_info.child_group_idxes;
    child_num=length(child_group_idxes);
    child_valid_flags=data_info.ref.child_valid_flags;
    assert(any(child_valid_flags));

        
        init_group_output_info=data_info.ref.output_info_layers{end};
                       
        assert(init_group_output_info.is_group_data);
        output_info_child_groups=init_group_output_info.data_child_groups;
        output_child_valid_flags=init_group_output_info.child_valid_flags;
        init_bp_child_valid_flags=init_group_output_info.bp_child_valid_flags;
        
        assert(length(output_info_child_groups)==child_num);
        assert(length(output_child_valid_flags)==child_num);
        
        
        first_output_info=data_info.ref.output_info_layers{1};
        assert(first_output_info.is_group_data);
                   
        bp_child_valid_flags=false(child_num, 1);
        for g_idx=1:child_num
                                    
            child_valid=output_child_valid_flags(g_idx);
            if ~child_valid
                continue;
            end
            
            init_bp_finished=init_bp_child_valid_flags(g_idx);
            if ~init_bp_finished
                continue;
            end
            
            child_group_idx=child_group_idxes(g_idx);
            child_init_output_info=output_info_child_groups{g_idx};
            
            if ~check_valid_net_output(child_init_output_info)
                continue;
            end
            
            child_first_output_info=do_backward_child_group( net_config,...
                work_info_batch, child_group_idx, child_init_output_info);
            
            first_output_info.data_child_groups{g_idx}=child_first_output_info;
                       
            bp_child_valid_flags(g_idx)=child_first_output_info.bp_finished;
           
        end
        
        
        first_output_info.bp_child_valid_flags=bp_child_valid_flags;
        first_output_info.bp_finished=any(bp_child_valid_flags);
               

        data_info.ref.output_info_layers{1}=first_output_info;
end







function do_backward_one_group( net_config, work_info_batch, group_idx)




data_info=work_info_batch.ref.data_info_groups{group_idx};
if ~data_info.ref.need_bp
    return;
end

group_info=net_config.ref.group_infos{group_idx};


if ~group_info.skip_backward
    
    
    child_group_idxes=group_info.child_group_idxes;


    if isempty(child_group_idxes)

        net_run_config=work_info_batch.ref.net_run_config;
                        
        backward_time=tic;
        do_bp_current_net=check_do_bp_current_net(work_info_batch, net_run_config, ...
            retrieve_net_info(group_info, net_config));
        if do_bp_current_net
            
            work_info_batch.ref.current_group_data_info=data_info;
            
            my_net_backward(retrieve_net_info(group_info, net_config), ...
                work_info_batch, data_info, net_run_config);
            work_info_batch.ref.leaf_group_bp_flags(group_idx)=true;
            
            work_info_batch.ref.current_group_data_info=[];
            
        end
        
        backward_time=toc(backward_time);
        work_info_batch.ref.backward_time=work_info_batch.ref.backward_time+backward_time;

    else

        switch group_info.child_relation
            case 'chain'
                do_backward_group_child_chain(net_config, work_info_batch, data_info, group_info);
            case 'parallel'
                do_backward_group_child_parallel(net_config, work_info_batch, data_info, group_info);
            otherwise
                error('not support!');
        end

    end
end


data_info.ref.output_info_layers=data_info.ref.output_info_layers(1);


end




function first_output_info=do_backward_child_group( ...
    net_config, work_info_batch, child_group_idx, init_output_info)
    
    child_data_info=work_info_batch.ref.data_info_groups{child_group_idx};
    
    if ~child_data_info.ref.need_bp
        first_output_info=[];
        first_output_info.bp_finished=false;
        return;
    end
    
    if ~check_valid_net_output(init_output_info)
        first_output_info=[];
        first_output_info.bp_finished=false;
        return;
    end
   
    child_data_info.ref.output_info_layers{end}=init_output_info;
        do_run_one_group(net_config, work_info_batch, child_group_idx, 'backward');
    
    first_output_info=child_data_info.ref.output_info_layers{1};
    if isempty(first_output_info)
        first_output_info.bp_finished=false;
    end
        
    child_data_info.ref=[];
    
end





function one_net_info=retrieve_net_info(group_info, net_config)

one_net_info=group_info.net_info;
if isempty(one_net_info)
    return;
end
if isempty(one_net_info.ref.layers)
    linked_group_info=net_config.ref.group_infos{one_net_info.ref.linked_group_idx};
    one_net_info=linked_group_info.net_info;
end

end



