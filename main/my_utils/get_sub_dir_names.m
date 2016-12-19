
function dir_names=get_sub_dir_names(root_dir)


file_infos=dir(root_dir);
dir_names=[];
dir_dates=[];

for f_idx=1:length(file_infos)
   
    one_f_info=file_infos(f_idx);
    if one_f_info.isdir
        dir_name=one_f_info.name;
        
        if ~strcmp(dir_name(1), '.')
            dir_names=cat(1, dir_names, {dir_name});
            dir_dates=cat(1, dir_dates, datenum(one_f_info.date));
        end
    end
    
end

% sort by create date:
[~, tmp_sort_idxes]=sort(dir_dates);
dir_names=dir_names(tmp_sort_idxes);


end