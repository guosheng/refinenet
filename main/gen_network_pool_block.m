
function layer_gen_info=gen_network_pool_block(refine_config, dag_net, layer_gen_info, layer_name_prefix)

    if ~refine_config.use_chained_pool
        return;
    end

    % original implmentation:
	% layer_gen_info=do_gen_block(refine_config, dag_net, layer_gen_info, layer_name_prefix);
		
    % improved version:
	layer_gen_info=do_gen_block_convbeforepool(refine_config, dag_net, layer_gen_info, layer_name_prefix);

end




function [outputs, one_output_dim]=add_joint_layer_dagnn(dag_net, inputs, name, use_concat, joint_input_dims)


outputs={[name '_varout']};

if use_concat
    block = My_concat_layer() ;
else
    block = My_sum_layer() ;
end

assert(isa(inputs, 'cell'));
if size(inputs, 1)>1
    inputs=inputs';
    joint_input_dims=joint_input_dims';
end
block.numInputs=length(inputs);

dag_net.addLayer(...
    name, ...
    block, ...
    inputs, ...
    outputs, ...
    {}) ;


if use_concat
            
    one_output_dim=sum(joint_input_dims);
    
    feat_dim_before=one_output_dim;
   

    assert(all(joint_input_dims==joint_input_dims(1)));
    feat_dim_after=joint_input_dims(1);

    outputs=My_net_util.add_dim_reduce_layer(dag_net, outputs{1}, feat_dim_before, feat_dim_after);

    one_output_dim=feat_dim_after;
    
else
    assert(joint_input_dims(1)==joint_input_dims(2));
    one_output_dim=joint_input_dims(1);
end
      
  
end







function layer_gen_info=do_gen_block(refine_config, dag_net, layer_gen_info, layer_name_prefix)

	
    one_outputs=layer_gen_info.one_outputs;
    one_outputs=My_net_util.add_relu_dagnn(dag_net, [layer_name_prefix '_poolprev'], one_outputs);
    layer_gen_info.one_outputs=one_outputs;


    pool_num=refine_config.chained_pool_num;
    pool_size=refine_config.chained_pool_size;

    assert(pool_num>=1);

    one_output_dim=layer_gen_info.one_output_dim;
        
    one_outputs=layer_gen_info.one_outputs;
    pool_outputs=one_outputs;
       
                
    for p_idx=1:pool_num
        
        block = dagnn.Pooling() ;
        block.method = 'max' ;
        block.poolSize = [pool_size pool_size] ;
        block.pad =  floor(pool_size./2); 
        block.stride = 1 ;
                
        layer_name=sprintf([layer_name_prefix '_pool%d'], p_idx);
        
        one_inputs=one_outputs;
        
        one_outputs={[layer_name, '_outvar']};
        
        dag_net.addLayer(...
            layer_name, ...
            block, ...
            one_inputs, ...
            one_outputs, ...
            {}) ;
        
        
        feat_dim_before=one_output_dim;
        feat_dim_after=one_output_dim;
        one_outputs=My_net_util.add_dim_reduce_layer(dag_net, one_outputs{1}, feat_dim_before, feat_dim_after);
        
        pool_outputs=cat(2, pool_outputs, one_outputs);
        
    end
    
    
    
        use_concat=false;
    
        joint_layer_name=[layer_name_prefix '_pool_joint'];
        one_inputs=pool_outputs;
        joint_input_dims=repmat(one_output_dim, [1, pool_num+1]);
        [one_outputs, joint_output_dim]=add_joint_layer_dagnn(dag_net, one_inputs, joint_layer_name, use_concat, joint_input_dims);

        
    layer_gen_info.one_output_dim=joint_output_dim;
    layer_gen_info.one_outputs=one_outputs;
        

end







function layer_gen_info=do_gen_block_convbeforepool(refine_config, dag_net, layer_gen_info, layer_name_prefix)

	
    one_outputs=layer_gen_info.one_outputs;
    one_outputs=My_net_util.add_relu_dagnn(dag_net, [layer_name_prefix '_poolprev'], one_outputs);
    layer_gen_info.one_outputs=one_outputs;


    pool_num=refine_config.chained_pool_num;
    pool_size=refine_config.chained_pool_size;

    assert(pool_num>=1);

    one_output_dim=layer_gen_info.one_output_dim;
        
    one_outputs=layer_gen_info.one_outputs;
    pool_outputs=one_outputs;


                
    for p_idx=1:pool_num


    	feat_dim_before=one_output_dim;
        feat_dim_after=one_output_dim;

        dim_adapt_name=sprintf([one_outputs{1} '_pb%d'], p_idx);
        one_outputs=My_net_util.add_dim_reduce_layer_named(dag_net, one_outputs{1}, feat_dim_before, feat_dim_after, dim_adapt_name);

        
        block = dagnn.Pooling() ;
        block.method = 'max' ;
        block.poolSize = [pool_size pool_size] ;
        block.pad =  floor(pool_size./2); 
        block.stride = 1 ;
                
        layer_name=sprintf([layer_name_prefix '_pool%d'], p_idx);
        
        one_inputs=one_outputs;
        
        one_outputs={[layer_name, '_outvar']};
        
        dag_net.addLayer(...
            layer_name, ...
            block, ...
            one_inputs, ...
            one_outputs, ...
            {}) ;
                        
        
        pool_outputs=cat(2, pool_outputs, one_outputs);
        
    end
    
    
    
        use_concat=false;
    
        joint_layer_name=[layer_name_prefix '_pool_joint'];
        one_inputs=pool_outputs;
        joint_input_dims=repmat(one_output_dim, [1, pool_num+1]);
        [one_outputs, joint_output_dim]=add_joint_layer_dagnn(dag_net, one_inputs, joint_layer_name, use_concat, joint_input_dims);

      
        
        
    layer_gen_info.one_output_dim=joint_output_dim;
    layer_gen_info.one_outputs=one_outputs;
        

end









