






function group_output_info=gen_network_refine_group(train_opts, net_config, group_input_info)



multipath_fusion_group_info=My_net_util.gen_group_info_basic(net_config, 'multipath_fusion');


layers=cell(0);

[one_layers, net_input_info]=gen_input_path_merge_net(train_opts, group_input_info);
layers=cat(2, layers, one_layers);


dag_net=dagnn.DagNN();
[one_layers, net_output_info]=gen_refinenet(dag_net, train_opts.refine_config, net_input_info);
layers=cat(2, layers, one_layers);


group_output_info=net_output_info;

net_info=My_net_util.gen_net_info_basic(train_opts);
net_info.ref.name='multipath_fusion';
net_info.ref.layers=layers;

multipath_fusion_group_info.net_info=net_info;
multipath_fusion_group_info.dag_net=dag_net;
if ~isempty(dag_net)
    multipath_fusion_group_info.use_dagnn=true;
end

net_config.ref.group_infos{multipath_fusion_group_info.group_idx,1}=multipath_fusion_group_info;
net_config.ref.root_group_info.child_group_idxes(end+1,1)=multipath_fusion_group_info.group_idx;


end






function [layers, net_input_info]=gen_input_path_merge_net(train_opts, group_input_info)


layers=cell(0);

one_layer=[];
one_layer.type='my_custom';
one_layer.custom_type='input_concat';

one_layer.forward_fn=@cnn_layer_path_merge_forward;
one_layer.backward_fn=@cnn_layer_path_merge_backward;

one_layer.layer_update_fn=[];
one_layer.layer_result_name='inputpath_fusion';
layers{end+1}=one_layer;

input_info_paths=group_input_info.child_infos;
input_var_names=cell(0);
input_dim_vars=zeros(0);
for p_idx=1:length(input_info_paths)
    one_input_info=input_info_paths{p_idx};
    input_var_names=cat(1, input_var_names, one_input_info.var_names);
    input_dim_vars=cat(1, input_dim_vars, one_input_info.var_dims);
end


% replace the var names:
refine_config=train_opts.refine_config;
refine_config_levels=refine_config.refine_config_levels;
level_num=length(refine_config_levels);
assert(length(input_var_names)==level_num)
for l_idx=1:level_num
    input_var_names{l_idx}=refine_config_levels{l_idx}.input_name;
end

net_input_info=[];
net_input_info.var_names=input_var_names;
net_input_info.var_dims=input_dim_vars;


end






function [layers, net_output_info]=gen_refinenet(dag_net, refine_config, net_input_info)


net_input_info_org=net_input_info;

net_input_info=add_adapt_block_stages(refine_config, dag_net, net_input_info);



net_output_info=do_gen_refinenet_stage_group(dag_net, refine_config, net_input_info);


My_net_util.fix_padding_resnet(dag_net);


% debug:
% dag_net.print('Format', 'dot')


layers=cell(0);

one_layer=[];
one_layer.type='my_custom';
one_layer.custom_type='dagnn_wrapper';
one_layer.forward_fn=@cnn_layer_dagnn_wrapper_forward;
one_layer.backward_fn=@cnn_layer_dagnn_wrapper_backward;
one_layer.layer_update_fn=[];


% should be careful of the orders of var_names, should not directly use dag_net.getInputs();
% use net_input_info instead

one_layer.input_var_names=net_input_info_org.var_names;
tmp_inpus=dag_net.getInputs();
assert(all(ismember(tmp_inpus, net_input_info_org.var_names)));
assert(length(tmp_inpus)==length(net_input_info_org.var_names));


stageout_num=length(net_output_info.var_names);
assert(stageout_num==1);
one_layer.output_var_names=cell(stageout_num, 1);
one_layer.output_var_names{1}=net_output_info.var_names{1};
one_layer.use_single_output=stageout_num==1;

tmp_outpus=dag_net.getOutputs();
assert(all(ismember(tmp_outpus, net_output_info.var_names)));



layers{end+1}=one_layer;


end









function [outputs, one_output_dim]=add_joint_layer_dagnn(dag_net, inputs, name, use_concat, joint_input_dims, init_lr)


outputs={[name '_varout']};

if use_concat
    block = My_concat_layer() ;
else
    block = My_sum_layer() ;
end

block.numInputs=2;

dag_net.addLayer(...
    name, ...
    block, ...
    inputs, ...
    outputs, ...
    {}) ;


if use_concat
    
    assert(~isempty(init_lr));
    one_output_dim=sum(joint_input_dims);
    feat_dim_before=one_output_dim;
    assert(all(joint_input_dims==joint_input_dims(1)));
    feat_dim_after=joint_input_dims(1);

    outputs=My_net_util.add_dim_reduce_layer(dag_net, outputs{1}, feat_dim_before, feat_dim_after, init_lr);

    one_output_dim=feat_dim_after;
    
else
    assert(joint_input_dims(1)==joint_input_dims(2));
    one_output_dim=joint_input_dims(1);
end
      
  
end






function [new_branch_var_name, new_branch_var_dim]=add_stage_adapt_block(...
            refine_config, dag_net, input_var_name, var_output_dim, init_lr, stage_idx)

    one_input_dim=var_output_dim;
    one_inputs={input_var_name};
        
    refine_dim=refine_config.refine_config_levels{stage_idx}.input_adapt_dim;
    one_output_dim=refine_dim;
    
    one_outputs=My_net_util.add_dim_reduce_layer(dag_net, one_inputs{1}, one_input_dim, one_output_dim, init_lr);
    
    layer_gen_info=[];
    layer_gen_info.one_outputs=one_outputs;
    layer_gen_info.one_output_dim=one_output_dim;
    layer_gen_info.init_lr=init_lr;

    layer_name_prefix=sprintf('adapt_stage%d', stage_idx);
    
    conv_num=refine_config.adapt_conv_num;
    assert(conv_num>0);
    if conv_num>0
        layer_gen_info=My_net_util.add_res_conv_block(dag_net, layer_gen_info, one_output_dim, conv_num, layer_name_prefix);
    end
    

      
        
    one_outputs=layer_gen_info.one_outputs;
    one_output_dim=layer_gen_info.one_output_dim;
        
    new_branch_var_dim=one_output_dim;
    new_branch_var_name=one_outputs{1};
    
end



function net_output_info=add_adapt_block_stages(refine_config, dag_net, net_input_info)
    
    var_names=net_input_info.var_names;
    valid_output_dim_vars=net_input_info.var_dims;    
    
    init_lr=refine_config.lr_factor;
    
    for stage_idx=1:length(var_names)
        
        conn_var_name=var_names{stage_idx};
        [stage_output_name, stage_output_dim]=add_stage_adapt_block(...
            refine_config, dag_net, conn_var_name, valid_output_dim_vars(stage_idx), init_lr, stage_idx);
        
         var_names{stage_idx}=stage_output_name;
        valid_output_dim_vars(stage_idx)=stage_output_dim;
    end

    net_output_info=[];
    net_output_info.var_names=var_names;
    net_output_info.var_dims=valid_output_dim_vars;    
    
end











function net_output_info=do_gen_refinenet_joint(dag_net, refine_config, net_input_info, name_subfix)
   
    init_lr=refine_config.lr_factor;
    var_names=net_input_info.var_names;
    valid_output_dim_vars=net_input_info.var_dims;    
   
    filter_num_after_joint=refine_config.filter_num_after_joint;
    stage_num=length(var_names);
    

    if stage_num>1
	    for stage_idx=1:stage_num
	        
	        one_inputs=var_names(stage_idx);
	        one_input_dim=valid_output_dim_vars(stage_idx);
	                        
	        one_output_dim=filter_num_after_joint;
	        one_outputs=My_net_util.add_dim_reduce_layer(dag_net, one_inputs{1}, one_input_dim, one_output_dim, init_lr);
	                                
	        var_names(stage_idx)=one_outputs;
	        valid_output_dim_vars(stage_idx)=one_output_dim;
	    end
	    
	    
	    one_inputs=var_names;
	    joint_input_dims=valid_output_dim_vars;
	    layer_name_prefix=['mflow_joint' name_subfix];
	    [one_outputs, joint_output_dim]=add_joint_layer_dagnn(dag_net, one_inputs, layer_name_prefix, false, joint_input_dims, []);

	else

		one_outputs=var_names;
	    joint_output_dim=valid_output_dim_vars;
	    assert(filter_num_after_joint==joint_output_dim);

	end

    mainflow_layer_gen_info=[];
    mainflow_layer_gen_info.one_outputs=one_outputs;
    mainflow_layer_gen_info.one_output_dim=joint_output_dim;
    mainflow_layer_gen_info.init_lr=init_lr;

	layer_name_prefix=['mflow_conv' name_subfix];            
        
    mainflow_layer_gen_info=gen_network_pool_block(refine_config, dag_net, mainflow_layer_gen_info, layer_name_prefix);
    
    conv_num=refine_config.refine_block_conv_num_mainflow;
    if conv_num>0
        one_output_dim=mainflow_layer_gen_info.one_output_dim;
        mainflow_layer_gen_info=My_net_util.add_res_conv_block(dag_net, mainflow_layer_gen_info, one_output_dim, conv_num, layer_name_prefix);
    end


    
    net_output_info=[];
    net_output_info.var_names=mainflow_layer_gen_info.one_outputs;
    net_output_info.var_dims=mainflow_layer_gen_info.one_output_dim;

end







function net_output_info=do_gen_refinenet_stage_group(dag_net, refine_config, net_input_info)
  
    
    var_names=net_input_info.var_names;
    var_dims=net_input_info.var_dims;    
    
    
    group_ids=refine_config.group_ids;
    if isempty(group_ids)
        group_size=refine_config.group_size;
        assert(group_size>=1);
        
        level_num=length(var_names);
        group_ids=[];
            
        can_level_idxes=1:level_num;
        id_counter=0;
        while ~isempty(can_level_idxes)
            id_counter=id_counter+1;
            
            one_ids=repmat(id_counter, [1, group_size-1]);
        	
            if length(one_ids)>length(can_level_idxes)
                one_ids=one_ids(1:length(can_level_idxes));
            end
            group_ids=cat(2, group_ids, one_ids);
            can_level_idxes=can_level_idxes(length(one_ids)+1:end);
        end
    end
    
    group_id_values=unique(group_ids);
    group_num=length(group_id_values);
        
            
    prev_output_info=[];
    
    for g_idx=1:group_num
        
        member_flags=group_ids==group_id_values(g_idx);
        
        
        one_var_names=var_names(member_flags);
        one_var_dims=var_dims(member_flags);
        if ~isempty(prev_output_info)
            one_var_names=cat(1, prev_output_info.var_names, one_var_names);
            one_var_dims=cat(1, prev_output_info.var_dims, one_var_dims);
        end
        one_net_input_info=[];
        one_net_input_info.var_names=one_var_names;
        one_net_input_info.var_dims=one_var_dims;
        
        one_refine_config=refine_config;
        one_refine_config.filter_num_after_joint=min(one_var_dims);
               
                
        prev_output_info=do_gen_refinenet_joint(dag_net, one_refine_config, one_net_input_info, ['_g' num2str(g_idx)]);
        
    end
    
    net_output_info=prev_output_info;
    
end




