



function loss_group_info=gen_network_loss_group(train_opts, net_config, group_input_info)


feat_output_dim=group_input_info.var_dims;
assert(length(feat_output_dim)==1);

loss_group_info=My_net_util.gen_group_info_basic(net_config, 'loss_group');

stageout_info=[];
stageout_info.stage_idx=1;
stageout_info.output_dim=feat_output_dim;
stageout_info.stageout_num=1;
stageout_info.gen_prediction_info=true;
           

[loss_group_info.net_info, dag_net_lossgroup]=do_gen_net_info(train_opts, stageout_info);     

if ~isempty(dag_net_lossgroup)
    loss_group_info.dag_net=dag_net_lossgroup;
    loss_group_info.use_dagnn=true;
end

loss_group_info.forward_evaluate_fn=@cnn_loss_group_evaluate;

loss_group_info.prediction_layer_idxes=loss_group_info.net_info.ref.prediction_layer_idxes;
net_config.ref.group_infos{loss_group_info.group_idx,1}=loss_group_info;

net_config.ref.root_group_info.child_group_idxes(end+1,1)=loss_group_info.group_idx;


end






function [net_info, dag_net]=do_gen_net_info(train_opts, stageout_info)

    loss_config=train_opts.loss_config;
    class_num=loss_config.class_num;
    loss_group_input_dim=stageout_info.output_dim;
            
    layers=cell(0);
          
    
    dag_net=[];
    if loss_config.lossgroup_conv_num>0
        [dag_net, output_dim_crf_conv1, one_layers]=gen_dag_net_lossgroup(loss_config, loss_group_input_dim);
        layers=cat(2, layers, one_layers);
    else
        output_dim_crf_conv1=loss_group_input_dim;
    end
    
    
    layers{end+1} = struct('type', 'dropout', 'rate', 0.5) ;

    
    output_dim_crf_conv2=class_num;
    input_dim_crf_conv2=output_dim_crf_conv1;
    layers{end+1} = struct('type', 'conv', ...
                   'filters', 0.01 * randn(3,3,input_dim_crf_conv2, output_dim_crf_conv2,'single'), ...
                   'biases', zeros(1, output_dim_crf_conv2, 'single'), ...
                   'stride', 1, ...
                   'pad', 0 );
                   

    one_layer=[];
    one_layer.type='my_custom';
    one_layer.custom_type='dense_softmaxloss';
    
    one_layer.forward_fn=@cnn_layer_dense_softmaxloss_forward;
    one_layer.backward_fn=@cnn_layer_dense_softmaxloss_backward;
   
    
    one_layer.stageout_info=stageout_info;
        
            
    one_layer.layer_update_fn=[];
    layers{end+1}=one_layer;
    
    
layers=gen_padding_keep_size(layers);


net_info=My_net_util.gen_net_info_basic();
                    

net_info.ref.name='final_loss';
net_info.ref.layers=layers;
net_info.ref.prediction_layer_idxes=length(layers);

net_info.ref.lr_multiplier=loss_config.lr_multiplier;

end



function [dag_net, output_dim, layers]=gen_dag_net_lossgroup(loss_config, loss_group_input_dim)

conv_num=loss_config.lossgroup_conv_num;
one_output_dim=loss_config.lossgroup_conv_filter_num;


dag_net=dagnn.DagNN();

start_var_name='data_input';

    
layer_gen_info=[];
layer_gen_info.one_output_dim=loss_group_input_dim;
layer_gen_info.one_outputs={start_var_name};
   
if loss_group_input_dim~=one_output_dim
	error('should not come here!');
end

layer_name_prefix=sprintf('lossblock');
layer_gen_info=My_net_util.add_res_conv_block(dag_net, layer_gen_info, one_output_dim, conv_num, layer_name_prefix);

output_dim=layer_gen_info.one_output_dim;

output_var_name=layer_gen_info.one_outputs{1};
output_var_idx=dag_net.getVarIndex(output_var_name);
dag_net.vars(output_var_idx).fanout=0;

% debug:
% dag_net.print('Format', 'dot');


My_net_util.fix_padding_resnet(dag_net);

layers=cell(0);

one_layer=[];
one_layer.type='my_custom';
one_layer.custom_type='dagnn_wrapper';
one_layer.forward_fn=@cnn_layer_dagnn_wrapper_forward;
one_layer.backward_fn=@cnn_layer_dagnn_wrapper_backward;
one_layer.layer_update_fn=[];

one_layer.input_var_names=dag_net.getInputs();
one_layer.output_var_names=dag_net.getOutputs();

assert(length(one_layer.input_var_names)==1);
assert(length(one_layer.output_var_names)==1);

one_layer.use_single_output=true;

layers{end+1}=one_layer;



end







