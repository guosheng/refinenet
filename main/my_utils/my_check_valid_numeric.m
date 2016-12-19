

function my_check_valid_numeric(feat_data)


try
    
    tmp_check=any(isnan(feat_data)) | any(isinf(feat_data));
    while numel(tmp_check)>1
        tmp_check=any(tmp_check);
    end
        
    assert(~gather(tmp_check));

        
catch err_info
    disp('my_check_valid_numeric failed');
    dbstack;
    keyboard;
end



end