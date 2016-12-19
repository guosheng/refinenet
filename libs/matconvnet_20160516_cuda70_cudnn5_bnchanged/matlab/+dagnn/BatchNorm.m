classdef BatchNorm < dagnn.ElementWise
  properties
    numChannels
    epsilon = 1e-4
    bnorm_moment_type_trn
    bnorm_moment_type_tst
    noise_param_idx
    noise_cache_size
  end

  methods
      
      
      function [input_data, scale_value]=add_noise_input(obj, inputs, params)
          
          error('not here!');
          
            scale_value=1;
            input_data=inputs{1};
            tmp_param_idx=obj.noise_param_idx;
            if ~isempty(params{tmp_param_idx})
                tmp_mean=params{tmp_param_idx};
                if size(tmp_mean, 2)>1
                    tmp_mean=tmp_mean(:, 1);
                end
                tmp_data=reshape(tmp_mean, [1 1 numel(tmp_mean)]);
                
                scale_value=0.5;
                input_data=bsxfun(@plus, input_data, tmp_data).*scale_value;
                
            end 
            
      end
      
      
      function [input_data]=adapt_input(obj, inputs, params)
          
          error('not here!');
                      
            input_data=inputs{1};
            tmp_param_idx=3;
            if ~isempty(params{tmp_param_idx})
                tmp_mean=params{tmp_param_idx};
                if size(tmp_mean, 2)>1
                    tmp_mean=tmp_mean(:, 1);
                end
                tmp_data=reshape(tmp_mean, [1 1 numel(tmp_mean)]);
                
                batch_mean=mean(input_data, 1);
                batch_mean=mean(batch_mean, 2);
                batch_mean=mean(batch_mean, 4);
                
                tmp_data=tmp_data-batch_mean;
                                
                input_data=bsxfun(@plus, input_data, tmp_data);
                
            end 
            
      end
      
      
      
    function outputs = forward(obj, inputs, params)
        
      if strcmp(obj.net.mode, 'test')
          do_trn=false;
          bnorm_moment_type=obj.bnorm_moment_type_tst;
      else
          do_trn=true;
          bnorm_moment_type=obj.bnorm_moment_type_trn;
      end
      
        if strcmp(bnorm_moment_type, 'batch_noise')
            
            error('not here!');
            assert(do_trn);
                        
            input_data=obj.add_noise_input(inputs, params);
            
            outputs{1} = vl_nnbnorm(input_data, params{1}, params{2}, ...
                                'epsilon', obj.epsilon) ;
                            
                                        
        elseif strcmp(bnorm_moment_type, 'batch')
            
            outputs{1} = vl_nnbnorm(inputs{1}, params{1}, params{2}, ...
                                'epsilon', obj.epsilon) ;
                            
                            
        elseif strcmp(bnorm_moment_type, 'global')
            
            outputs{1} = vl_nnbnorm(inputs{1}, params{1}, params{2}, ...
                                'moments', params{3}, ...
                                'epsilon', obj.epsilon) ;
            
                            
        else
            error('not support!');
        end
     
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
        
                
        bnorm_moment_type=obj.bnorm_moment_type_trn;
        if strcmp(bnorm_moment_type, 'batch_noise')
            
            error('not here!');
            
            [input_data, scale_value]=obj.add_noise_input(inputs, params);
            output_der=derOutputs{1};
            
            [derInputs{1}, derParams{1}, derParams{2}, derParams{3}] = ...
                vl_nnbnorm(input_data, params{1}, params{2}, output_der, ...
                           'epsilon', obj.epsilon) ;
            
            tmp_mean=derParams{3}(:,1);
            derParams{4}=tmp_mean;
            derInputs{1}=derInputs{1}.*scale_value;
                                   
            
        elseif strcmp(bnorm_moment_type, 'batch')
            
            [derInputs{1}, derParams{1}, derParams{2}, derParams{3}] = ...
                vl_nnbnorm(inputs{1}, params{1}, params{2}, derOutputs{1}, ...
                           'epsilon', obj.epsilon) ;
                                   
            %debug:
%             moments=derParams{3};
%             [der1, der2, der3] = vl_nnbnorm(inputs{1}, params{1}, params{2}, derOutputs{1}, ...
%                             'moments', moments, ...
%                             'epsilon', obj.epsilon) ;
%             check_der1=derInputs{1};
%             % if  this two value is the same, then the gradient is not
%             % correct, and not treat moement as a constant!!!
%             disp(max(abs(der1(:)-check_der1(:))));
              
                                      
        elseif strcmp(bnorm_moment_type, 'global')
            
                [derInputs{1}, derParams{1}, derParams{2}, derParams{3}] = ...
                    vl_nnbnorm(inputs{1}, params{1}, params{2}, derOutputs{1}, ...
                           'moments', params{3}, ...
                           'epsilon', obj.epsilon) ;
                      
        else
            error('not support!');
        end
        
        % multiply the moments update by the number of images in the batch
          % this is required to make the update additive for subbatches
          % and will eventually be normalized away
          input_num=size(inputs{1},4);
          derParams{3} = derParams{3} * input_num ;
              
    end

    % ---------------------------------------------------------------------
    function obj = BatchNorm(varargin)
      obj.load(varargin{:}) ;
    end

    function params = initParams(obj)
      params{1} = ones(obj.numChannels,1,'single') ;
      params{2} = zeros(obj.numChannels,1,'single') ;
      params{3} = zeros(obj.numChannels,2,'single') ;
    end

    function attach(obj, net, index)
      attach@dagnn.ElementWise(obj, net, index) ;
      p = net.getParamIndex(net.layers(index).params{3}) ;
      net.params(p).trainMethod = 'average' ;
      net.params(p).learningRate = 0.01 ;
    end
  end
end
