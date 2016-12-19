



function my_clear_net_trn(net_config)

    cnn_update_net_group(net_config, @do_clear, []);

end

function one_group_info=do_clear(one_group_info, opts)


net_info=one_group_info.net_info;
if isempty(net_info)
    return;
end


for i=1:numel(net_info.ref.layers)


  if strcmp(net_info.ref.layers{i}.type,'conv')
      
      
      net_info.ref.layers{i}.filtersMomentum = zeros('like',net_info.ref.layers{i}.filters) ;
      net_info.ref.layers{i}.biasesMomentum = zeros('like',net_info.ref.layers{i}.biases) ;
            
     
      if ~isfield(net_info.ref.layers{i}, 'filtersLearningRate')
        net_info.ref.layers{i}.filtersLearningRate = 1 ;
      end
      if ~isfield(net_info.ref.layers{i}, 'biasesLearningRate')
        net_info.ref.layers{i}.biasesLearningRate = 1 ;
      end
      if ~isfield(net_info.ref.layers{i}, 'filtersWeightDecay')
        net_info.ref.layers{i}.filtersWeightDecay = 1 ;
      end
      if ~isfield(net_info.ref.layers{i}, 'biasesWeightDecay')
        net_info.ref.layers{i}.biasesWeightDecay = 1 ;
      end
  
  end
  
  
  if strcmp(net_info.ref.layers{i}.type,'my_custom')
      layer_update_fn=net_info.ref.layers{i}.layer_update_fn;
      if ~isempty(layer_update_fn)
          update_info=[];
          update_info.group_info=one_group_info;
          net_info.ref.layers{i}=net_info.ref.layers{i}.layer_update_fn(...
                'init_trn', net_info.ref.layers{i}, update_info);
      end
  end
  
end


end
