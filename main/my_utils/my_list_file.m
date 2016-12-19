
function [file_names, file_infos]=my_list_file(ds_dir)


file_infos=dir(ds_dir);
file_names=cell(0, 1);
valid_sel=false(length(file_infos), 1);
for file_idx=1:length(file_infos)
    
    if file_infos(file_idx).isdir
        continue;
    end
    
    one_file_name=file_infos(file_idx).name;
    
    file_names{file_idx}=one_file_name;
    valid_sel(file_idx)=true;
end

file_names=file_names(valid_sel);
file_infos=file_infos(valid_sel);


end