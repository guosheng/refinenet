
function flag=my_any(one_exclude_flags)

    flag=any(one_exclude_flags);
    if(numel(flag)>1)
        flag=my_any(flag);
    end
end