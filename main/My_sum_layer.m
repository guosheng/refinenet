classdef My_sum_layer < dagnn.ElementWise
  %SUM DagNN sum layer
  %   The SUM layer takes the sum of all its inputs and store the result
  %   as its only output.

  properties (Transient)
    numInputs=[];
  end

  properties
    use_gradient_scale_factor=true;
  end
  
  methods(Static)
            
  end

  methods
      
      function [one_x, gradient_factor]=do_resize(obj, target_size, one_x)
          feat_map_size=size(one_x);
          feat_map_size=feat_map_size(1:2);
          gradient_factor=[];
           if any(target_size~=feat_map_size)
                                
                one_x=my_resize(one_x, target_size);
                
                assert(~isempty(obj.use_gradient_scale_factor));
                if obj.use_gradient_scale_factor
                    gradient_factor=prod(feat_map_size)/prod(target_size);
                end
           end
      end
      
    function outputs = forward(obj, inputs, params)
      obj.numInputs = numel(inputs) ;
      
      target_size=size(inputs{1});
      for k = 2:obj.numInputs
          target_size=max(target_size, size(inputs{k}));
      end
      target_size=target_size(1:2);
      
      outputs{1} = obj.do_resize(target_size, inputs{1});
      
      for k = 2:obj.numInputs
          one_map2=obj.do_resize(target_size, inputs{k});
          outputs{1} = outputs{1} + one_map2;
      end
      
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
        
      one_der=derOutputs{1};
%       current_size=size(one_der);
%       current_size=current_size(1:2);
        
      for k = 1:obj.numInputs
         
         target_size=size(inputs{k});
         target_size=target_size(1:2);
         [new_der, gradient_factor]=obj.do_resize(target_size, one_der);
         if ~isempty(gradient_factor)
             new_der=new_der.*gradient_factor;
         end
         derInputs{k} = new_der;
         new_der=[];
         
      end
      derParams = {} ;
    end

    function outputSizes = getOutputSizes(obj, inputSizes)
        
      outputSizes{1} = inputSizes{1} ;
      
      for k = 2:numel(inputSizes)
        if all(~isnan(inputSizes{k})) && all(~isnan(outputSizes{1}))
            
            map_size1=inputSizes{k}(1:2);
            map_size2=outputSizes{1}(1:2);
          
            outputSizes{1}(1:2)=max(map_size2, map_size1);
            outputSizes{1}(3)=min(inputSizes{k}(3), outputSizes{1}(3));
        end
      end
    end

    function rfs = getReceptiveFields(obj)
      numInputs = numel(obj.net.layers(obj.layerIndex).inputs) ;
      rfs.size = [1 1] ;
      rfs.stride = [1 1] ;
      rfs.offset = [1 1] ;
      rfs = repmat(rfs, numInputs, 1) ;
    end

    function obj = My_sum_layer(varargin)
      obj.load(varargin) ;
    end
  end
end
