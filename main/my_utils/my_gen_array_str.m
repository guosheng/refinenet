
function array_str=my_gen_array_str(values, num_pattern)

    if nargin<2
        num_pattern='%d';
    end

    array_str='[';
    for f_idx=1:length(values)
        if f_idx>1
            array_str=cat(2, array_str, ',');
        end
        array_str=cat(2, array_str, sprintf(num_pattern, values(f_idx)));
    end
    array_str=[array_str ']'];
    
    
end