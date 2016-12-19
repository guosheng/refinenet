

function x=move_to_gpu_values(x)

if iscell(x)
    for tmp_idx=1:length(x)
        x{tmp_idx}=gpuArray(x{tmp_idx});
    end
else
    x=gpuArray(x);
end

end