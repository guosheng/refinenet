




function node_scores=my_logsoftmax(node_scores)
  

if size(node_scores, 1)>1

    node_scores=bsxfun(@minus, node_scores, max(node_scores, [], 2));
    tmp_node_scores=node_scores;
    node_scores=exp(node_scores);
    node_scores=bsxfun(@minus, tmp_node_scores,...
        log(sum(node_scores, 2)));
else
    node_scores=node_scores-max(node_scores);
    tmp_node_scores=node_scores;
    node_scores=exp(node_scores);
    node_scores=tmp_node_scores-log(sum(node_scores));
end


end