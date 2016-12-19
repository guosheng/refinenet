
function [dir_names, file_infos]=my_list_dir(ds_dir)


file_infos=dir(ds_dir);
dir_names=cell(0, 1);
valid_sel=false(length(file_infos), 1);
for dir_idx=1:length(file_infos)
    one_dir_name=file_infos(dir_idx).name;
    if strcmp(one_dir_name, '.') || strcmp(one_dir_name, '..')
       continue; 
    end
    if ~file_infos(dir_idx).isdir
        continue;
    end
    dir_names{dir_idx}=one_dir_name;
    valid_sel(dir_idx)=true;
end

dir_names=dir_names(valid_sel);
file_infos=file_infos(valid_sel);


end