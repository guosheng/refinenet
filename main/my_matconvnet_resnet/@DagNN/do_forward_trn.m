
%Author: Guosheng Lin


function do_forward_trn(obj, inputs)

%copy from  matconvnet/matlab/+dagnn/@DagNN/eval.m
% copy this function to matconvnet/matlab/+dagnn/@DagNN/


obj.computingDerivative = true;


% -------------------------------------------------------------------------
% Forward pass
% -------------------------------------------------------------------------

% set the input values
v = obj.getVarIndex(inputs(1:2:end)) ;
if any(isnan(v))
  broken = find(isnan(v)) ;
  error('No variable of name ''%s'' could be found in the DAG.', inputs{2*broken(1)-1}) ;
end
[obj.vars(v).value] = deal(inputs{2:2:end}) ;
inputs = [] ;

obj.numPendingVarRefs = [obj.vars.fanout] ;
for l = obj.executionOrder
  time = tic ;
  obj.layers(l).block.forwardAdvanced(obj.layers(l)) ;
  obj.layers(l).forwardTime = toc(time) ;
end



end
