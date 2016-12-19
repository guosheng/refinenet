function file_exist=my_check_file(cache_file, be_quite)

if nargin<2
    be_quite=false;
end


if exist(cache_file, 'file')
    
    file_exist=true;
    
    if ~be_quite
        fprintf('file exist:%s\n',cache_file);
    end
else

    file_exist=false;
    
    if ~be_quite
        fprintf('file not exist:%s\n',cache_file);
    end
end




end

