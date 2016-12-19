function mkdir_notexist(one_dir)

if ~exist(one_dir,'dir')
    unix(['mkdir -p ' one_dir]);
end

end