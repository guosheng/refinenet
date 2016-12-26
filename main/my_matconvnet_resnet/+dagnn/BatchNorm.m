classdef BatchNorm < dagnn.ElementWise
  properties
    numChannels
    epsilon = 1e-5
    opts = {'NoCuDNN'} % ours seems slightly faster
    noise_param_idx
    noise_cache_size
    bnorm_moment_type_trn
    bnorm_moment_type_tst
  end

  properties (Transient)
    moments
  end

  methods
    function outputs = forward(obj, inputs, params)

	  if strcmp(obj.net.mode, 'test')
        outputs{1} = vl_nnbnorm(inputs{1}, params{1}, params{2}, ...
                                'moments', params{3}, ...
                                'epsilon', obj.epsilon, ...
                                obj.opts{:}) ;
      else

      	% Guosheng: changed for RefineNet
      	bnorm_moment_type=obj.bnorm_moment_type_trn;

        if strcmp(bnorm_moment_type, 'batch')

	        [outputs{1},obj.moments] = ...
	            vl_nnbnorm(inputs{1}, params{1}, params{2}, ...
	                       'epsilon', obj.epsilon, ...
	                       obj.opts{:}) ;
	            
		elseif strcmp(bnorm_moment_type, 'global')
			outputs{1} = vl_nnbnorm(inputs{1}, params{1}, params{2}, ...
                                'moments', params{3}, ...
                                'epsilon', obj.epsilon, ...
                                obj.opts{:}) ;
		else
            error('not support!');
        end

      end



    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)

		% Guosheng: changed for RefineNet
		
    	bnorm_moment_type=obj.bnorm_moment_type_trn;

    	if strcmp(bnorm_moment_type, 'batch')

	      [derInputs{1}, derParams{1}, derParams{2}, derParams{3}] = ...
	        vl_nnbnorm(inputs{1}, params{1}, params{2}, derOutputs{1}, ...
	                   'epsilon', obj.epsilon, ...
	                   'moments', obj.moments, ...
	                   obj.opts{:}) ;
	      

  		elseif strcmp(bnorm_moment_type, 'global')

  			[derInputs{1}, derParams{1}, derParams{2}, derParams{3}] = ...
	        	vl_nnbnorm(inputs{1}, params{1}, params{2}, derOutputs{1}, ...
	        		'moments', params{3}, ...
	                   'epsilon', obj.epsilon, ...
	                   obj.opts{:}) ;	

	    else
            error('not support!');
  		end
  	
  		obj.moments = [] ;


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
      net.params(p).learningRate = 0.1 ;
    end
  end
end
