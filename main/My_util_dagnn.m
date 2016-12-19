classdef My_util_dagnn

properties

end

methods(Static)

% -------------------------------------------------------------------------
    function layers = simpleFindLayersOfType(net, type)
    % -------------------------------------------------------------------------
    layers = find(cellfun(@(x)strcmp(x.type, type), net.layers)) ;
    end

    % -------------------------------------------------------------------------
    function net = simpleRemoveLayersOfType(net, type)
    % -------------------------------------------------------------------------
    layers = simpleFindLayersOfType(net, type) ;
    net.layers(layers) = [] ;
    end

    % -------------------------------------------------------------------------
    function layers = dagFindLayersWithOutput(net, outVarName)
    % -------------------------------------------------------------------------
    layers = {} ;
    for l = 1:numel(net.layers)
      if any(strcmp(net.layers(l).outputs, outVarName))
        layers{1,end+1} = net.layers(l);
      end
    end
    
    end

    % -------------------------------------------------------------------------
    function layers = dagFindLayersOfType(net, type)
    % -------------------------------------------------------------------------
    layers = [] ;
    for l = 1:numel(net.layers)
      if isa(net.layers(l).block, type)
        layers{1,end+1} = net.layers(l).name ;
      end
    end
    
    end

    % -------------------------------------------------------------------------
    function dagRemoveLayersOfType(net, type)
    % -------------------------------------------------------------------------
    names = dagFindLayersOfType(net, type) ;
    for i = 1:numel(names)
      layer = net.layers(net.getLayerIndex(names{i})) ;
      net.removeLayer(names{i}) ;
      net.renameVar(layer.outputs{1}, layer.inputs{1}, 'quiet', true) ;
    end
    
    end
    
    
    % -------------------------------------------------------------------------
    function [layers, layer_idxes] = dagFindLayersWithInput(net, varName)
    % -------------------------------------------------------------------------
    layers = {} ;
    layer_idxes=[];
    for l = 1:numel(net.layers)
      if any(strcmp(net.layers(l).inputs, varName))
        layers{1,end+1} = net.layers(l) ;
        layer_idxes(1, end+1)=l;
      end
    end
    
    end
    
    
    
    function f = bilinear_u(k, numGroups, numClasses)
    %BILINEAR_U  Create bilinear interpolation filters
    %   BILINEAR_U(K, NUMGROUPS, NUMCLASSES) compute a square bilinear filter
    %   of size k for deconv layer of depth numClasses and number of groups
    %   numGroups

    factor = floor((k+1)/2) ;
    if rem(k,2)==1
      center = factor ;
    else
      center = factor + 0.5 ;
    end
    C = 1:k ;
    if numGroups ~= numClasses
      f = zeros(k,k,numGroups,numClasses) ;
    else
      f = zeros(k,k,1,numClasses) ;
    end

    for i =1:numClasses
      if numGroups ~= numClasses
        index = i ;
      else
        index = 1 ;
      end
      f(:,:,index,i) = (ones(1,k) - abs(C-center)./factor)'*(ones(1,k) - abs(C-center)./(factor));
    end
    end


    

end

end




