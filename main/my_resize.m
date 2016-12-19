

function x=my_resize(x, map_size)

% x=do_resize_cpu_only(x, map_size);

x=do_resize_with_gpu(x, map_size);

end


function x=do_resize_cpu_only(x, map_size)


map_size=double(map_size);
assert(length(map_size)==2);
x_size=[size(x,1) size(x,2)];
if all(x_size==map_size)
    return;
end

if isa(x, 'gpuArray')
	x=gpuArray(imresize(gather(x), map_size, 'bicubic'));    
else
    assert(isa(x, 'single'));
    x=imresize(x, map_size, 'bicubic');
end

end



function x=do_resize_with_gpu(x, map_size)

map_size=double(map_size);
assert(length(map_size)==2);
x_size=[size(x,1) size(x,2)];
if all(x_size==map_size)
    return;
end

if isa(x, 'gpuArray')
	
    one_scale=max(map_size./x_size);
    
    x=imresize(x, one_scale);
    new_size=size(x);
    new_size=new_size(1:2);
    if any(new_size~=map_size)

        if ~all(new_size>=map_size)
        	dbstack;
            keyboard;
		end
        
        x=x(1:map_size(1), 1:map_size(2), :);
        
        check_size=size(x);
        if ~all(check_size(1:2)==map_size)
            dbstack;
            keyboard;
        end
        
    end
else
    assert(isa(x, 'single'));
    x=imresize(x, map_size, 'bicubic');
end

end


