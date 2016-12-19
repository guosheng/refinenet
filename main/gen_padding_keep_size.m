

function layers=gen_padding_keep_size(layers)

    for l_idx=1:length(layers)
        l=layers{l_idx};
        if strcmp(l.type, 'conv')
            filter_size=size(l.filters);
            pad_size1=round((filter_size(1)-1)/2);
            pad_size2=round((filter_size(2)-1)/2);
            %To verify the order...
            l.pad=[pad_size1, pad_size2, pad_size1, pad_size2];
            
%             l.stride=[1 1];

        end
        if strcmp(l.type, 'pool')
            pool_size=l.pool;
            pad_size1=round((pool_size(1)-1)/2);
            pad_size2=round((pool_size(2)-1)/2);
            %To verify the order...
            l.pad=[pad_size1, pad_size2, pad_size1, pad_size2];
            
%             l.stride=[1 1];
        end
        layers{l_idx}=l;
    end

end
