

function [layers, net_output_info, dag_net]=gen_network_featnet_imgraw(featnet_config)


dag_net=dagnn.DagNN();
start_var_name='data_input';

layer_gen_info=[];
layer_gen_info.one_output_dim=3;
layer_gen_info.one_outputs={start_var_name};

layer_gen_info=do_add_img_input_scratch_path(dag_net, featnet_config, layer_gen_info);


net_output_info=[];
net_output_info.var_names=layer_gen_info.one_outputs;
net_output_info.var_dims=layer_gen_info.one_output_dim;


My_net_util.fix_padding_resnet(dag_net);


% input_vars=net.getInputs;
% dag_net.print({input_vars{1}, [400 400 3]}, 'Format', 'dot');


layers=cell(0);


one_layer=[];
one_layer.type='my_custom';
one_layer.custom_type='dagnn_wrapper';
one_layer.forward_fn=@cnn_layer_dagnn_wrapper_forward;
one_layer.backward_fn=@cnn_layer_dagnn_wrapper_backward;
one_layer.layer_update_fn=[];

one_layer.input_var_names=dag_net.getInputs();
one_layer.output_var_names=dag_net.getOutputs();

one_layer.use_single_output=true;

assert(length(one_layer.input_var_names)==1);
assert(length(one_layer.output_var_names)==1);
assert(all(ismember(net_output_info.var_names, one_layer.output_var_names)));


layers{end+1}=one_layer;



end








function layer_gen_info=do_add_img_input_scratch_path(dag_net, featnet_config, layer_gen_info)

layer_gen_info=add_scratch_img_input_block(dag_net, featnet_config, layer_gen_info);

conv_num=featnet_config.conv_num_one_block;
conv_block_num=featnet_config.conv_block_num;
for b_idx=1:conv_block_num
    
    one_output_dim=featnet_config.filter_num_blocks(b_idx);
    layer_name_prefix=sprintf('inputblock%d', b_idx);
    
    
    if b_idx>=2
        layer_gen_info=do_add_res_conv_block_downsample(dag_net, layer_gen_info, one_output_dim, conv_num, layer_name_prefix);
    else
        layer_gen_info=My_net_util.add_res_conv_block(dag_net, layer_gen_info, one_output_dim, conv_num, layer_name_prefix);
    end
    
end


output_var_name=layer_gen_info.one_outputs{1};
output_var_idx=dag_net.getVarIndex(output_var_name);
dag_net.vars(output_var_idx).fanout=0;

end









function layer_gen_info=add_scratch_img_input_block(dag_net, featnet_config, layer_gen_info)


input_dim=layer_gen_info.one_output_dim;
one_output_dim=featnet_config.filter_num_blocks(1);

filter_size=[3 3 input_dim one_output_dim]; 
one_inputs=layer_gen_info.one_outputs;
filter_name='imginput_c1';
one_outputs={[filter_name '_outvar']};
My_net_util.add_conv_layer_dagnn(dag_net, one_inputs, one_outputs, false, filter_size, filter_name, true);



tmp_l_idx=dag_net.getLayerIndex(filter_name);
dag_net.layers(tmp_l_idx).block.stride=2;


one_outputs=My_net_util.add_relu_dagnn(dag_net, filter_name, one_outputs);

layer_gen_info.one_output_dim=one_output_dim;
layer_gen_info.one_outputs=one_outputs;


filter_size=[3 3 one_output_dim one_output_dim]; 
one_inputs=layer_gen_info.one_outputs;
filter_name='imginput_c2';
one_outputs={[filter_name '_outvar']};
My_net_util.add_conv_layer_dagnn(dag_net, one_inputs, one_outputs, false, filter_size, filter_name, true);



layer_gen_info.one_outputs=one_outputs;


end







function layer_gen_info=do_add_res_conv_block_downsample(dag_net, layer_gen_info, output_dim, conv_num, layer_name_prefix)


one_output_dim=layer_gen_info.one_output_dim;
feat_dim_before=one_output_dim;
feat_dim_after=output_dim;
one_outputs_dimred=layer_gen_info.one_outputs;

[one_outputs_dimred, layer_name_dimred]=My_net_util.add_dim_reduce_layer(dag_net, one_outputs_dimred{1}, feat_dim_before, feat_dim_after);

layer_gen_info.one_output_dim=feat_dim_after;

tmp_l_idx=dag_net.getLayerIndex(layer_name_dimred);
dag_net.layers(tmp_l_idx).block.stride=2;

layer_gen_info.one_outputs=one_outputs_dimred;

layer_gen_info=My_net_util.add_res_conv_block(dag_net, layer_gen_info, output_dim, conv_num, layer_name_prefix);


end




