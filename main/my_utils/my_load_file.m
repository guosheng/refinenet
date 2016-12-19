

function tmp=my_load_file(cache_file, be_quite)

    if nargin<2
        be_quite=false;
    end

    if ~be_quite
        fprintf('load_file:%s\n',cache_file);
    end

        
    tmp=[];
    if exist(cache_file, 'file')
                
        finished=false;
        while ~finished
            try
                tmp=load(cache_file);
                finished=true;
            catch err_info
                disp(err_info);
                pause_sec=60+60*rand(1);
                fprintf('load_file failed, retry in %.f sec, file:%s\n', pause_sec, cache_file);
                pause(pause_sec);
            end
        end
    else
        
        error('file not found!');
    end

end


