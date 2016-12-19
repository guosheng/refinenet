


function my_move_net(net_info, destination)


if strcmp(destination, 'gpu')

    for l=1:numel(net_info.ref.layers)
      switch net_info.ref.layers{l}.type
        case 'conv'
          for f = {'filters', 'biases', 'filtersMomentum', 'biasesMomentum'}
            f = char(f) ;
            if isfield(net_info.ref.layers{l}, f)
              net_info.ref.layers{l}.(f) = move_to_gpu_values(net_info.ref.layers{l}.(f)) ;
            end
          end
        otherwise
          % nothing to do ?
      end
    end
    
    net_info.ref.net_on_gpu=true;

end


if strcmp(destination, 'cpu')

    for l=1:numel(net_info.ref.layers)
      switch net_info.ref.layers{l}.type
        case 'conv'
          for f = {'filters', 'biases', 'filtersMomentum', 'biasesMomentum'}
            f = char(f) ;
            if isfield(net_info.ref.layers{l}, f)
              net_info.ref.layers{l}.(f) = move_to_cpu_values(net_info.ref.layers{l}.(f)) ;
            end
          end
        otherwise
          % nothing to do ?
      end
    end
    
    net_info.ref.net_on_gpu=false;

end




end
