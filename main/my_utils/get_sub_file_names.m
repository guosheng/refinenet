
function file_names=get_sub_file_names(root_dir)


file_infos=dir(root_dir);
file_names=[];
file_dates=[];

for f_idx=1:length(file_infos)
   
    one_f_info=file_infos(f_idx);
    if ~one_f_info.isdir
        file_name=one_f_info.name;
        file_names=cat(1, file_names, {file_name});
        file_dates=cat(1, file_dates, datenum(one_f_info.date));
    end
    
end

% sort by create date:
[~, tmp_sort_idxes]=sort(file_dates);
file_names=file_names(tmp_sort_idxes);


end