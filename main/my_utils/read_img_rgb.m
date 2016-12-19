function img_data=read_img_rgb(img_file, data_type)

assert(strcmp(data_type, 'uint8'));

try

	[img_data, map] = imread(img_file);

catch err_info
	
	disp(err_info);
	disp(img_file);
	error('image read error');

end



if size(img_data, 4)>1
	
	error('cannot read images with multi-frames');	

    % only load the first frame
    img_data=img_data(:,:,:, 1);
end


if ~isempty(map)
	img_data = ind2rgb(img_data,map);
end


if size(img_data, 3)==1
               
    [img_data map]=gray2ind(img_data);
    img_data=ind2rgb(img_data, map);
    
end



if strcmp(data_type, 'uint8')
    img_data=im2uint8(img_data);    
end
if strcmp(data_type, 'double')
    img_data=im2double(img_data);    
end
if strcmp(data_type, 'single')
    img_data=im2single(img_data);    
end


try 
    assert(size(img_data, 3)==3);
catch error_msg
    disp(error_msg);
    dbstack;
    keyboard;
end


end


