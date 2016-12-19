


function input_info = cnn_layer_dagnn_wrapper_backward(input_info, layer, work_info_batch, output_info)



group_idx=work_info_batch.ref.net_run_current_group_idx;

% the group_info maybe the linked group_info, don't use this group_idx
group_info=get_current_work_group_idx(work_info_batch);
net=group_info.dag_net;



% input_var_name=layer.input_var_name;
% input_data={input_var_name, input_info.x};



output_num=length(layer.output_var_names);

if output_info.is_group_data
    assert(~layer.use_single_output);
    derOutputs=cell(0);
    for o_idx=1:output_num
        derOutputs = cat(2, derOutputs, { layer.output_var_names{o_idx}, output_info.data_child_groups{o_idx}.dzdx} );
        assert(~isempty(output_info.data_child_groups{o_idx}.dzdx));
    end
else
    assert(layer.use_single_output);
    derOutputs={layer.output_var_names{end}, output_info.dzdx};
    assert(~isempty(output_info.dzdx));
end


% derOutputs={layer.output_var_name, output_info.dzdx};


net.mode = 'normal' ;
net.do_backward_trn(derOutputs) ;


% accumulate gradient
work_info=work_info_batch.ref.work_info;
state=work_info.ref.tmp_cache.dag_state_groups{group_idx};


net_info=group_info.net_info;
current_step=work_info_batch.ref.epoch;
lr_steps=net_info.ref.lr_steps;
input_lr=lr_steps(min(length(lr_steps), current_step));
state.learningRate=input_lr;


batchSize=1;
state = accumulate_gradients(state, net, batchSize) ;
work_info.ref.tmp_cache.dag_state_groups{group_idx}=state;


input_var_names=layer.input_var_names;
input_num=length(input_var_names);
input_var_idxes=zeros(input_num,1);
for in_idx=1:input_num
    input_var_idxes(in_idx)=net.getVarIndex(input_var_names{in_idx});
end
assert(all(input_var_idxes>0));


if input_info.is_group_data
    input_num=length(layer.input_var_names);
    for v_idx=1:input_num
        one_der=net.vars(input_var_idxes(v_idx)).der;
        input_info.data_child_groups{v_idx}.dzdx=one_der;
        assert(~isempty(one_der));
    end
else
    assert(length(input_var_idxes)==1);
    dzdx=net.vars(input_var_idxes(1)).der;
    input_info.dzdx=dzdx;
    assert(~isempty(dzdx));
end

end








function state = accumulate_gradients(state, net, batchSize)
% -------------------------------------------------------------------------

for p=1:numel(net.params)

  % accumualte gradients from multiple labs (GPUs) if needed

  switch net.params(p).trainMethod

    case 'average' % mainly for batch normalization
        % verify: actually this value is not considered in the training
        % process, so accumulate this just for testing....
      
      if isempty(net.params(p).value)
        net.params(p).value=net.params(p).der;
      else
        thisLR = net.params(p).learningRate ;
        net.params(p).value = ...
          (1 - thisLR) * net.params(p).value + ...
          (thisLR/batchSize/net.params(p).fanout) * net.params(p).der ;
      end
      

    case 'gradient'
      thisDecay = state.weightDecay * net.params(p).weightDecay ;
      thisLR = state.learningRate * net.params(p).learningRate ;
      
%       disp(net.params(p))
%       disp(thisLR)
      
      state.momentum{p} = state.momentum_param * state.momentum{p} ...
        - thisDecay * net.params(p).value ...
        - (1 / batchSize) * net.params(p).der ;
        
      net.params(p).value = net.params(p).value + thisLR * state.momentum{p} ;

    case 'fixed'
        %do nothing
%         disp('debug');
        
    case 'otherwise'
      error('Unknown training method ''%s'' for parameter ''%s''.', ...
        net.params(p).trainMethod, ...
        net.params(p).name) ;
  end
end



end