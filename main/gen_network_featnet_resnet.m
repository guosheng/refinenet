

function [layers, net_output_info, dag_net]=gen_network_featnet_resnet(featnet_config)


pre_trained_model_file=featnet_config.pre_trained_model_file;

fprintf('load init model:%s\n', pre_trained_model_file);
tmp_obj=load(pre_trained_model_file) ;

dag_net = dagnn.DagNN.loadobj(tmp_obj) ;
dag_net.removeLayer('prob');
dag_net.removeLayer('fc1000');
dag_net.removeLayer('pool5');

My_net_util.fix_batch_norm_resnet(dag_net);
   
net_output_info=gen_net_output_info_resnet(dag_net, featnet_config);
var_names=net_output_info.var_names;
var_dims=net_output_info.var_dims;
for stage_idx=1:length(var_names)

    conn_var_name=var_names{stage_idx};
    if var_dims(stage_idx)>512
        conn_var_name=add_init_net_dropout(dag_net, conn_var_name);
    else
        conn_var_name=My_net_util.add_var_copy_dagnn(dag_net, conn_var_name, conn_var_name);
        conn_var_name=conn_var_name{1};
    end
    var_names{stage_idx}=conn_var_name;
    conn_var_idx=dag_net.getVarIndex(var_names{stage_idx});
    assert(dag_net.vars(conn_var_idx).fanout==0);
end
net_output_info.var_names=var_names;

My_net_util.fix_padding_resnet(dag_net);

layers=cell(0);

one_layer=[];
one_layer.type='my_custom';
one_layer.custom_type='dagnn_wrapper';
one_layer.forward_fn=@cnn_layer_dagnn_wrapper_forward;
one_layer.backward_fn=@cnn_layer_dagnn_wrapper_backward;
one_layer.layer_update_fn=[];

tmp_inputs=dag_net.getInputs();
assert(length(tmp_inputs)==1);
one_layer.input_var_names=tmp_inputs;


% for the order problem, should not directly use dag_net.getOutputs()
tmp_outputs=dag_net.getOutputs();
assert(length(net_output_info.var_names)==length(tmp_outputs));
assert(all(ismember(tmp_outputs, net_output_info.var_names)));

one_layer.output_var_names=net_output_info.var_names;
one_layer.use_single_output=false;

layers{end+1}=one_layer;


end
















function net_output_info=gen_net_output_info_resnet(dag_net, featnet_config)


init_stage_idx=featnet_config.init_output_path_idx;
output_path_num=featnet_config.output_path_num;

tmp_net_output_info=gen_stage_output_info(dag_net, featnet_config);


var_names=tmp_net_output_info.var_names(init_stage_idx:end);
valid_output_dim_vars=tmp_net_output_info.valid_output_dim_vars(init_stage_idx:end);
   

if init_stage_idx>1
    target_var_name=var_names{1};
    [~, target_layer_idxes] = My_util_dagnn.dagFindLayersWithInput(dag_net, target_var_name);
    run_l_idxes=dag_net.getLayerExecutionOrder;
    tmp_flags=ismember(run_l_idxes, target_layer_idxes);
    tmp_remove_idx=find(tmp_flags, 1);
    to_remove_layer_idxes=run_l_idxes(tmp_remove_idx:end);
    
    to_remove_layer_names=cell(length(to_remove_layer_idxes), 1);
    for l_idx=1:length(to_remove_layer_idxes)
        to_remove_layer_names{l_idx}=dag_net.layers(to_remove_layer_idxes(l_idx)).name;
    end
    
    for l_idx=1:length(to_remove_layer_idxes)
        dag_net.removeLayer(to_remove_layer_names{l_idx});
    end
    
end
   
net_output_info=[];
net_output_info.var_names=var_names(1:output_path_num);
net_output_info.var_dims=valid_output_dim_vars(1:output_path_num);


end




function net_output_info=gen_stage_output_info(dag_net, featnet_config)


    var_num=length(dag_net.vars);
    stage_var_flags=false(var_num, 1);
    
    output_dim_vars=zeros(var_num, 1);
        
    input_var_name=dag_net.getInputs();
    assert(length(input_var_name)==1);
    input_var_name=input_var_name{1};
    input_var_idx=dag_net.getVarIndex(input_var_name);
    stage_var_flags(input_var_idx)=true;
    
    start_conv_layer=My_util_dagnn.dagFindLayersWithInput(dag_net,input_var_name);
    assert(length(start_conv_layer)==1);
    start_conv_layer=start_conv_layer{1};
    input_var_dim=start_conv_layer.block.size(3);
    output_dim_vars(input_var_idx)=input_var_dim;
           
    
    layer_num=length(dag_net.layers);
    layers_org=dag_net.layers;
    for l_idx=1:layer_num
        l=layers_org(l_idx);
        one_block=l.block;
        if isa(one_block, 'dagnn.Conv')
            first_var_dim=l.block.size(4);
        end
        if isa(one_block, 'dagnn.ReLU')
            start_v_idx=dag_net.getVarIndex(l.outputs{1});
            break;
        end
    end
    
    
    invalid_var_names={dag_net.vars(start_v_idx).name};
    
    for v_idx=start_v_idx:var_num
        
        if dag_net.vars(v_idx).fanout>1
            var_name=dag_net.vars(v_idx).name;
            conn_layers = My_util_dagnn.dagFindLayersWithInput(dag_net, var_name);
            assert(length(conn_layers)==2);
            
            one_var_dim=[];
            is_down_sample=true;
            for cl_idx=1:length(conn_layers)
                one_l=conn_layers{cl_idx};
                one_block=one_l.block;
                if ~isa(one_block, 'dagnn.Conv')
                    is_down_sample=false;
                else
                    if isempty(one_var_dim)
                        one_var_dim=one_block.size(3);
                    else
                        assert(one_var_dim==one_block.size(3));
                    end
                end
            end
                        
            if is_down_sample
                                
                input_layers = My_util_dagnn.dagFindLayersWithOutput(dag_net, var_name);
                assert(length(input_layers)==1);
                
                if isa(input_layers{1}.block, 'dagnn.ReLU')
                    output_dim_vars(v_idx)=one_var_dim;
                    stage_var_flags(v_idx)=true;
                end
            end
            
        end
        
    end

    final_output_var_name=dag_net.getOutputs();
    final_var_idx=dag_net.getVarIndex(final_output_var_name);
    stage_var_flags(final_var_idx)=true;
    output_dim_vars(final_var_idx)=2048;
    
    var_idxes=find(stage_var_flags);
    var_idxes=var_idxes(end:-1:1);
    
    var_names=cell(length(var_idxes), 1);
    for v_idx=1:length(var_names)
        var_names{v_idx}=dag_net.vars(var_idxes(v_idx)).name;
    end
    
    valid_output_dim_vars=output_dim_vars(var_idxes);
    
    if ~isempty(invalid_var_names)
        tmp_valid_flags=~ismember(var_names, invalid_var_names);
        var_names=var_names(tmp_valid_flags);
        valid_output_dim_vars=valid_output_dim_vars(tmp_valid_flags);
    end
        
    % notes: don't use the var idx, use the var name instead
    net_output_info=[];
    net_output_info.var_names=var_names;
    net_output_info.valid_output_dim_vars=valid_output_dim_vars;
        
    % for resnet:
    assert(length(var_names)==5);
    
    assert(all(net_output_info.valid_output_dim_vars>0));
      
    
end


function tmp_var_name=add_init_net_dropout(dag_net, start_var_name)

tmp_var_name=My_net_util.add_dropout_dagnn(dag_net, start_var_name, {start_var_name});
tmp_var_name=tmp_var_name{1};
tmp_var_idx=dag_net.getVarIndex(tmp_var_name);
assert(dag_net.vars(tmp_var_idx).fanout==0);

end




