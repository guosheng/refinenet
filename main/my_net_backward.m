

function my_net_backward(net_info, work_info_batch, data_info, net_run_config)

    gpu_mode=net_info.ref.use_gpu;
      
    if gpu_mode
      if ~net_info.ref.net_on_gpu
           my_move_net(net_info, 'gpu') ;
      end
      data_info.ref.output_info_layers{end}=...
          move_output_info_gpu(data_info.ref.output_info_layers{end});
    else
      assert(~net_info.ref.net_on_gpu);
      data_info.ref.output_info_layers{end}=...
          move_output_info_cpu(data_info.ref.output_info_layers{end});
    end
    
    do_backward(net_info, work_info_batch, data_info, net_run_config);
    
    if gpu_mode
        if ~net_info.ref.net_stay_on_gpu
            my_move_net(net_info, 'cpu') ;
        end
    end
      
end




function do_backward(net_info, work_info_batch, data_info, net_run_config)


current_step=work_info_batch.ref.epoch;

lr_steps=net_info.ref.lr_steps;
input_lr=lr_steps(min(length(lr_steps), current_step));


net_info.ref.current_lr=input_lr;


bp_start_layer=net_info.ref.bp_start_layer;
layer_num = numel(net_info.ref.layers) ;
assert(bp_start_layer<=layer_num);


gpu_mode=net_run_config.use_gpu;

if gpu_mode && net_run_config.sync
    wait(gpuDevice) ;
end


if gpu_mode
    if ~net_info.ref.data_stay_on_gpu
        data_info.ref.output_info_layers{end}=move_output_info_gpu(...
            data_info.ref.output_info_layers{end});
    end
end
    


for layer_idx=layer_num:-1:bp_start_layer
    
    input_info=data_info.ref.output_info_layers{layer_idx};
    output_info=data_info.ref.output_info_layers{layer_idx+1};
        
    if gpu_mode && ~net_info.ref.data_stay_on_gpu
        input_info=move_output_info_gpu(input_info);
    end

    l = net_info.ref.layers{layer_idx} ;
    is_simple_layer= ~strcmp(l.type, 'my_custom');
    
    if is_simple_layer
        assert(~input_info.is_group_data);
        assert(~output_info.is_group_data);
        switch l.type
          case 'conv'

            one_dzdw=cell(2, 1);
            [input_info.dzdx, one_dzdw{1}, one_dzdw{2}] = ...
                vl_nnconv(input_info.x, l.filters, l.biases, ...
                          output_info.dzdx, ...
                          'pad', l.pad, 'stride', l.stride) ;
            input_info.dzdw=one_dzdw;

          case 'pool'
            input_info.dzdx = vl_nnpool(input_info.x, l.pool, output_info.dzdx, ...
              'pad', l.pad, 'stride', l.stride, 'method', l.method) ;
          case 'normalize'
            input_info.dzdx = vl_nnnormalize(input_info.x, l.param, output_info.dzdx) ;
          case 'softmax'
            input_info.dzdx = vl_nnsoftmax(input_info.x, output_info.dzdx) ;
          case 'relu'
            input_info.dzdx = vl_nnrelu(input_info.x, output_info.dzdx) ;
          case 'noffset'
            input_info.dzdx = vl_nnnoffset(input_info.x, l.param, output_info.dzdx) ;
          case 'dropout'
              input_info.dzdx = vl_nndropout(input_info.x, output_info.dzdx, 'mask', output_info.aux) ;
        end
    else
        if ~check_valid_net_output(output_info)
            break;
        end
        input_info = l.backward_fn(input_info, l, work_info_batch, output_info);
    end
    
    
    if gpu_mode && net_run_config.sync
      wait(gpuDevice) ;
    end
    
    input_info.bp_finished=true;
    data_info.ref.output_info_layers{layer_idx}=input_info;
    do_bp_update_one_layer(net_info, work_info_batch, input_info, layer_idx, input_lr);
        
    data_info.ref.output_info_layers{layer_idx+1}=[];

    if gpu_mode && net_run_config.sync
      wait(gpuDevice) ;
    end
end



end


 




function do_bp_update_one_layer(net_info, work_info_batch, input_info, layer_idx, input_lr)

    bp_start_layer=net_info.ref.bp_start_layer;
    if layer_idx<bp_start_layer
        return;
    end
           

      ly = net_info.ref.layers{layer_idx} ;
      lr=input_lr;

      if isfield(ly, 'update_layer_config_fn') && ~isempty(ly.update_layer_config_fn)
          ly.lr=lr;
          ly=ly.update_layer_config_fn(ly, net_info, work_info_batch, data_info);
          lr=ly.lr;
      end
      

      if strcmp(ly.type, 'conv') 

            momentum_param=net_info.ref.momentum;
            weightDecay_param=net_info.ref.weightDecay;

          ly.filtersMomentum = momentum_param * ly.filtersMomentum ...
              - weightDecay_param * ly.filtersWeightDecay ...
                  * lr * ly.filtersLearningRate * ly.filters ...
              - lr * ly.filtersLearningRate * input_info.dzdw{1} ;

          ly.biasesMomentum = momentum_param * ly.biasesMomentum ...
              - weightDecay_param * ly.biasesWeightDecay ...
                  * lr * ly.biasesLearningRate * ly.biases ...
              - lr * ly.biasesLearningRate * input_info.dzdw{2} ;

          ly.filters = ly.filters + ly.filtersMomentum ;
          ly.biases = ly.biases + ly.biasesMomentum ;
          net_info.ref.layers{layer_idx} = ly ;
      end


      if strcmp(ly.type, 'my_custom') 
          layer_update_fn=ly.layer_update_fn;
          if ~isempty(layer_update_fn)

              update_info=[];
              update_info.input_info=input_info;
              update_info.work_info_batch=work_info_batch;

              ly=layer_update_fn('bp_update', ly, net_info, update_info);
              net_info.ref.layers{layer_idx} = ly ;
          end
      end


end




