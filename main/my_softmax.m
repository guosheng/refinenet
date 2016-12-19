



function node_scores=my_softmax(node_scores)
  

if size(node_scores, 1)>1

    node_scores=bsxfun(@minus, node_scores, max(node_scores, [], 2));
    node_scores=exp(node_scores);
    node_scores=bsxfun(@rdivide, node_scores,...
        sum(node_scores, 2));
else
    node_scores=node_scores-max(node_scores);
    node_scores=exp(node_scores);
    node_scores=node_scores./sum(node_scores);
end


end