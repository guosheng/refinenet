

function my_imshow(img_data)

try
    imshow(img_data);
catch err_info
    disp('image cannot display!');
end

end