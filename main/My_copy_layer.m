classdef My_copy_layer < dagnn.ElementWise
  properties
  end


  methods
    function outputs = forward(obj, inputs, params)
        outputs = inputs ;
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
        derInputs = derOutputs ;
        derParams = {} ;
    end

    % ---------------------------------------------------------------------
    function obj = DropOut(varargin)
      obj.load(varargin{:}) ;
    end

    function obj = reset(obj)
      reset@dagnn.ElementWise(obj) ;
    end
    
  end
  
end
