
function tmp_str=cat_str(cellstrs)

tmp_str=[];
for c_idx=1:length(cellstrs)
    if ~isempty(tmp_str)
        tmp_str=[tmp_str '_'];
    end
    tmp_str=[tmp_str cellstrs{c_idx}];
end

end