
%modified by Guosheng Lin


classdef BatchNorm < dagnn.ElementWise
  properties
    numChannels
    epsilon = 1e-4
    bnorm_moment_type_trn
    bnorm_moment_type_tst
  end

  methods
    function outputs = forward(obj, inputs, params)
      if strcmp(obj.net.mode, 'test')
          bnorm_moment_type=obj.bnorm_moment_type_tst;
      else
          bnorm_moment_type=obj.bnorm_moment_type_trn;
      end
      
        if strcmp(bnorm_moment_type, 'batch')
            
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
        
        if strcmp(bnorm_moment_type, 'batch')
            
            [derInputs{1}, derParams{1}, derParams{2}, derParams{3}] = ...
                vl_nnbnorm(inputs{1}, params{1}, params{2}, derOutputs{1}, ...
                           'epsilon', obj.epsilon) ;
              
                            
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
          derParams{3} = derParams{3} * size(inputs{1},4) ;
      
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
