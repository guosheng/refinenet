

function x=move_to_cpu_values(x)

if iscell(x)
    for tmp_idx=1:length(x)
        x{tmp_idx}=gather(x{tmp_idx});
    end
else
    x=gather(x);
end

end