function my_save_image(one_img, one_file, be_quite)

    if nargin<3
        be_quite=false;
    end

    if ~be_quite
        fprintf('save_image, file:%s\n', one_file);
    end
    
    tmp_dir=fileparts(one_file);
    mkdir_notexist(tmp_dir);
    imwrite(one_img, one_file);
    
end