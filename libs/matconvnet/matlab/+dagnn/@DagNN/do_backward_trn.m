
%Author: Guosheng Lin

function do_backward_trn(obj, derOutputs)


% set output derivatives
derOutputsNames = derOutputs(1:2:end);
v = obj.getVarIndex(derOutputsNames) ;
if isnan(v)
  error('Invalid `derOutputs`, variables {%s} do not exist in the network.', ...
    strjoin(derOutputsNames(isnan(v)), ', '));
end
[obj.vars(v).der] = deal(derOutputs{2:2:end}) ;
derOutputs = [] ;

obj.numPendingVarRefs = zeros(1, numel(obj.vars)) ;
obj.numPendingParamRefs = zeros(1, numel(obj.params)) ;
for l = fliplr(obj.executionOrder)
  time = tic ;
  obj.layers(l).block.backwardAdvanced(obj.layers(l)) ;
  obj.layers(l).backwardTime = toc(time) ;
end

end
