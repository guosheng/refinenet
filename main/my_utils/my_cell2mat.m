

function b=my_cell2mat(a, cat_dim, e_count)


    
    if isempty(a)
        b=a;
        return;
    end
    
    cell_num=numel(a);
    if cell_num==1
        b=a{1};
        return;
    end
        
    if nargin<2
        cat_dim=[];
    end
    
    if nargin<3
        e_count=[];
    end
    
    if isempty(cat_dim)
        cat_dim=1;
    end
    
   
	assert(~isempty(a));
    
    temp_e=[];
       
    
    if isempty(e_count)
		e_count=0;
        for c_idx=1:cell_num
            if ~isempty(a{c_idx})
                one_size=size(a{c_idx}, cat_dim);
                e_count=e_count+one_size;
                if isempty(temp_e)
                    temp_e=a{c_idx};
                end
            end
        end
    else
        for c_idx=1:cell_num
            if ~isempty(a{c_idx})
                 temp_e=a{c_idx};
                 break;
            end
        end
    end

    
    if e_count==0 || isempty(temp_e)
        b=a{1};
        return;
    end
        
    new_mat_size=size(temp_e);
    new_mat_size(cat_dim)=e_count;
	    	
	if isnumeric(temp_e)

        % this will have problem when using gpuArray
%         class_name=class(temp_e);
% 		b=zeros(new_mat_size, class_name);
        
        b=zeros(new_mat_size, 'like', temp_e);
        
	elseif islogical(temp_e)
        
		b=false(new_mat_size);
        
	else
		error('not support element type!');
	end

    

    
    
    total_array_dim=numel(new_mat_size);
    assert(total_array_dim<=4);
        
    value_offset=0;
    for c_idx=1:cell_num
        one_mat=a{c_idx};
        if isempty(one_mat)
            continue;
        end
        value_start_idx=value_offset+1;
        value_end_idx=value_offset+size(one_mat, cat_dim);
        value_offset=value_end_idx;

        assert(value_offset<=e_count);
        
        if total_array_dim<=2
            if cat_dim==1
                b(value_start_idx:value_end_idx, :)=one_mat;
            end

            if cat_dim==2
                b(:, value_start_idx:value_end_idx)=one_mat;
            end
            
        else
            
            if cat_dim==1
                b(value_start_idx:value_end_idx, :, :,:)=one_mat;
            end

            if cat_dim==2
                b(:, value_start_idx:value_end_idx, :,:)=one_mat;
            end
            
            if cat_dim==3
                b(:, :, value_start_idx:value_end_idx,:)=one_mat;
            end
            
            if cat_dim==4
                b(:, :, :, value_start_idx:value_end_idx)=one_mat;
            end
            
        end

    end

    
    assert(value_offset==e_count);

end


