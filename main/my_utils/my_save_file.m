

function my_save_file(cache_file, data_obj, force_overide, be_quite)

if nargin<3
    force_overide=false;
end

if nargin<4
    be_quite=false;
end

finished=false;
while ~finished
    try
                
        if ~force_overide && exist(cache_file, 'file')
            disp('file exist, save to new name.');

            [f1, f2, f3]=fileparts(cache_file);
            while exist(cache_file, 'file')
                f2=[f2 '_new'];
                cache_file=[f1 '/' f2 f3];
            end

        end

        tmp_dir=fileparts(cache_file);
        mkdir_notexist(tmp_dir);

        if ~be_quite
            fprintf('save_file:%s\n',cache_file);
        end
        
        save(cache_file, 'data_obj', '-v7.3');
        finished=true;
    catch err_info
        disp(err_info);
        pause_sec=60+60*rand(1);
        fprintf('save_file failed, retry in %.f seconds, file:%s\n', pause_sec, cache_file);
        pause(pause_sec);
    end
end


end

