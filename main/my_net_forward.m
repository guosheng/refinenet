

        

function extra_data_info=my_net_forward(net_info, work_info_batch, data_info, net_run_config, extra_output_layer_idxes)

           
    gpu_mode=net_run_config.use_gpu;
    if gpu_mode
      if ~net_info.ref.net_on_gpu
           my_move_net(net_info, 'gpu') ;
      end
      data_info.ref.output_info_layers{1}=...
          move_output_info_gpu(data_info.ref.output_info_layers{1});
    else
      assert(~net_info.ref.net_on_gpu);
      data_info.ref.output_info_layers{1}=...
          move_output_info_cpu(data_info.ref.output_info_layers{1});
    end
    
    extra_data_info=do_forward(net_info, work_info_batch, data_info, net_run_config, extra_output_layer_idxes);
    
    if gpu_mode
        if ~net_info.ref.net_stay_on_gpu
            my_move_net(net_info, 'cpu') ;
        end
    end
    
       
end



function extra_data_info=do_forward(net_info, work_info_batch, data_info, net_run_config, extra_output_layer_idxes)


layer_num = numel(net_info.ref.layers) ;
gpu_mode = net_run_config.use_gpu;


if gpu_mode && net_run_config.sync
    wait(gpuDevice) ;
end

    

assert(check_valid_net_output(data_info.ref.output_info_layers{1}));


bp_turn_on_layer=-1;
disableDropout=true;
keep_layer_output=false;



do_bp_current_net=check_do_bp_current_net(work_info_batch, net_run_config, net_info);
if do_bp_current_net
    bp_start_layer=net_info.ref.bp_start_layer;
    bp_turn_on_layer=bp_start_layer;
end

data_info.ref.need_bp=do_bp_current_net;



extra_output_layer_flags=false(layer_num, 1);
if ~isempty(extra_output_layer_idxes)
    extra_output_layers=cell(layer_num, 1);
    extra_output_layer_flags(extra_output_layer_idxes)=true;
else
    extra_output_layers=[];
end




for layer_idx=1:layer_num
        
    input_info=data_info.ref.output_info_layers{layer_idx};
    l = net_info.ref.layers{layer_idx} ;
    is_simple_layer= ~strcmp(l.type, 'my_custom');
   
   
    if bp_turn_on_layer==layer_idx
        disableDropout=false;
        keep_layer_output=true;
    end
    
  
      

        if is_simple_layer
            output_info=data_info.ref.output_info_layers{layer_idx+1};
            assert(~input_info.is_group_data);
            assert(~output_info.is_group_data);
            switch l.type
                case 'conv'
                    input_size=size(input_info.x);
                    filter_size=size(l.filters);
                    if filter_size(3)~=input_size(3)
                        
                        disp('filter_size:');
                        disp(filter_size);
                        disp('input_size:');
                        disp(input_size);
                        error('filter size not match input size!');
                    end
                    if any(filter_size(1:2)>input_size(1:2))
                        error('filter size larger than the input size!');
                    end
                  output_info.x = vl_nnconv(input_info.x, l.filters, l.biases, 'pad', l.pad, 'stride', l.stride) ;
                case 'pool'
                    input_size=size(input_info.x);
                    pool_size=l.pool;
                    if any(pool_size(1:2)>input_size(1:2))
                        error('pool size larger than the input size!');
                    end
                    output_info.x = vl_nnpool(input_info.x, l.pool, 'pad', l.pad, 'stride', l.stride, 'method', l.method) ;
                case 'normalize'
                  output_info.x = vl_nnnormalize(input_info.x, l.param) ;
                case 'softmax'
                  output_info.x = vl_nnsoftmax(input_info.x) ;
                case 'relu'
                  output_info.x = vl_nnrelu(input_info.x) ;
                case 'noffset'
                  output_info.x = vl_nnnoffset(input_info.x, l.param) ;
                case 'dropout'
                  
                  if disableDropout
                    output_info.x = input_info.x ;
                  else
                    [output_info.x, output_info.aux] = vl_nndropout(input_info.x, 'rate', l.rate) ;
                  end

                otherwise
                    error('Unknown layer type %s', l.type) ;
            end
        else
            
            output_info= l.forward_fn(input_info, l, work_info_batch) ;
                       
            
        end

        if isempty(output_info)
            break;
        end
        output_info.forward_finished=true;

        
    if ~keep_layer_output && layer_idx < layer_num - 1
      data_info.ref.output_info_layers{layer_idx}=[];
      input_info=[];
    end

    if gpu_mode
        if ~net_info.ref.data_stay_on_gpu
            if ~isempty(input_info)
                input_info=move_output_info_cpu(input_info);
                data_info.ref.output_info_layers{layer_idx}=input_info;
            end
        end
    end
      
    if gpu_mode && net_run_config.sync
        wait(gpuDevice) ;
    end

    data_info.ref.output_info_layers{layer_idx+1}=output_info;
     
    if extra_output_layer_flags(layer_idx)
        extra_output_layers{layer_idx}=output_info;
    end
    
end


if gpu_mode && ~net_info.ref.data_stay_on_gpu
    output_info=data_info.ref.output_info_layers{end};
    output_info=move_output_info_cpu(output_info);
    data_info.ref.output_info_layers{end}=output_info;
end
       

extra_data_info=[];
extra_data_info.output_layers=extra_output_layers;


end



