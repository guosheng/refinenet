

function output_info=cnn_layer_dagnn_wrapper_forward(input_info, layer, work_info_batch)

   
run_trn=work_info_batch.ref.run_trn;
group_info=get_current_work_group_idx(work_info_batch);

net=group_info.dag_net;

if input_info.is_group_data
    input_var_names=layer.input_var_names;
    input_num=length(input_var_names);
    input_data=cell(0);
    for v_idx=1:input_num
        one_x=input_info.data_child_groups{v_idx}.x;
        assert(~isempty(one_x));
        input_data=cat(2, input_data, {input_var_names{v_idx}, one_x});
        assert(~isempty(one_x));
    end
else
    assert(length(layer.input_var_names)==1);
    input_var_name=layer.input_var_names{1};
    input_data={input_var_name, input_info.x};
    assert(~isempty(input_info.x));
end


do_bp_run=group_info.net_info.ref.do_bp && run_trn;

if do_bp_run
    
    net.mode = 'normal' ;
    net.do_forward_trn(input_data);
    
else
    
    net.mode = 'test' ;
    net.eval(input_data) ;
end


output_var_names=layer.output_var_names;
output_num=length(output_var_names);
output_var_idxes=zeros(output_num,1);
for o_idx=1:output_num
    output_var_idxes(o_idx)=net.getVarIndex(output_var_names{o_idx});
end
assert(all(output_var_idxes>0));


if layer.use_single_output
    output_x = net.vars(output_var_idxes(end)).value ;

    output_info=[];
    output_info.is_group_data=false;
    output_info.x=output_x;
    
    assert(~isempty(output_info.x));
else

    output_num=length(output_var_idxes);
    data_child_groups=cell(output_num, 1);
    for o_idx=1:output_num
        data_child_groups{o_idx}.is_group_data=false;
        data_child_groups{o_idx}.x = net.vars(output_var_idxes(o_idx)).value;
        
        assert(~isempty(data_child_groups{o_idx}.x));
    end

    output_info=[];
    output_info.is_group_data=true;
    output_info.data_child_groups=data_child_groups;
end


output_info=my_init_input_info(output_info);


% do the print here, generate the network graph :

% for resnet:
% input_vars=net.getInputs;
% net.print({input_vars{1}, [400 400 3]}, 'Format', 'dot');

% for cascaded refinenets
% input_vars=net.getInputs;
% net.print({input_vars{1}, [13 13 2048], input_vars{2}, [25 25 1024], input_vars{3}, [50 50 512], input_vars{4}, [100 100 256]}, 'Format', 'dot');


end




