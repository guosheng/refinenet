



function fuse_multiscale_results(fuse_param, class_info)


disp_step=20;

predict_result_dirs=fuse_param.predict_result_dirs;
fuse_result_dir=fuse_param.fuse_result_dir;
cache_fused_score_map=fuse_param.cache_fused_score_map;


disp('predict_result_dirs:');
for p_idx=1:length(predict_result_dirs)
    disp(predict_result_dirs{p_idx});
end


addpath('./my_utils');

score_dir_name='predict_result_full';
mask_dir_name='predict_result_mask';

one_score_dir=fullfile(predict_result_dirs{1}, score_dir_name);

file_infos=dir(fullfile(one_score_dir, '*.mat'));
assert(~isempty(file_infos));
task_num=length(file_infos);

fuse_score_dir=fullfile(fuse_result_dir, score_dir_name);
fuse_mask_dir=fullfile(fuse_result_dir, mask_dir_name);
mkdir_notexist(fuse_mask_dir);
if cache_fused_score_map
    mkdir_notexist(fuse_score_dir);
end


class_label_values=class_info.class_label_values;
mask_cmap=class_info.mask_cmap;


result_dir_num=length(predict_result_dirs);
% even weights:
result_weights=zeros(result_dir_num, 1)+single(1./result_dir_num);


for t_idx=1:task_num
    
    one_t=tic;
    
    
    one_score_file_name=file_infos(t_idx).name;
    [~, img_name, ~]=fileparts(one_score_file_name);
    one_mask_file_name=[img_name '.png'];

    fuse_score_file=fullfile(fuse_score_dir, one_score_file_name);
    fuse_mask_file=fullfile(fuse_mask_dir, one_mask_file_name);
    
        
    score_map=[];
    map_size=[];
    mask_size=[];

    for p_idx=1:result_dir_num
                
        one_result_dir=predict_result_dirs{p_idx};
        one_score_file=fullfile(one_result_dir, score_dir_name, one_score_file_name);
        one_score_map=do_load_score_map(one_score_file);
        assert(~isempty(one_score_map));
        
        one_result_weight=result_weights(p_idx);
        
        current_map_size=size(one_score_map);
        current_map_size=current_map_size(1:2);
        if isempty(score_map)
            score_map=one_score_map.*one_result_weight;
            map_size=current_map_size;

            one_mask_file=fullfile(one_result_dir, mask_dir_name, one_mask_file_name);
            one_mask_data=imread(one_mask_file);
            mask_size=size(one_mask_data);

            continue;            
        end


        if map_size(1)>=current_map_size(1)
            assert(map_size(2)>=current_map_size(2));
            one_score_map=do_score_map_resize(one_score_map, map_size);
        else
            map_size=current_map_size;
            score_map=do_score_map_resize(score_map, map_size);
        end
        
        score_map=score_map+one_score_map.*one_result_weight;
            
    end
       

    score_map=max(score_map, 0);
    score_map=min(score_map, 1);

        
    if cache_fused_score_map
        % notes: saved score map values range from 0 to 255
        cached_score_map=gather(im2uint8(score_map));
    end
        
    
    score_map=do_score_map_resize(score_map, mask_size);
    [~, predict_idx_mask]=max(score_map,[],3);
    predict_mask=class_label_values(predict_idx_mask);
    predict_mask=gather(predict_mask);

    assert(size(score_map, 3)<255);
    predict_mask=uint8(predict_mask);
    assert(all(size(predict_mask)==mask_size));
    imwrite(predict_mask, mask_cmap, fuse_mask_file);


    if cache_fused_score_map

        tmp_result_info=[];
        tmp_result_info.mask_data=predict_mask;
        tmp_result_info.score_map=cached_score_map;
        tmp_result_info.img_size=mask_size;
        tmp_result_info.class_info=class_info;
        my_save_file(fuse_score_file, tmp_result_info, true, true);          
    end

       
    if mod(t_idx, disp_step)==0 || t_idx==task_num || t_idx==1
        fprintf('fusing prediction, img:%d/%d, time:%.2f\n ---result_file:%s\n', ...
            t_idx, task_num, toc(one_t),  fuse_mask_file);
        
        my_diary_flush();
    end
end


end




function score_map=do_load_score_map(one_score_file)

    one_predict_result=my_load_file(one_score_file, true);
    one_predict_result=one_predict_result.data_obj;
        
    score_map=gather(one_predict_result.score_map);
        
    assert(isa(score_map, 'uint8'));
    score_map=single(score_map)./single(255);
   

end


function score_map=do_score_map_resize(score_map, target_size)
    
    score_map_size=size(score_map);
    score_map_size=score_map_size(1:2);
        
    if any(target_size~=score_map_size)
        
        score_map=log(score_map);
        score_map=max(score_map, -20);
        score_map=my_resize(score_map, target_size);
        score_map=exp(score_map);

%         score_map=my_resize(score_map, target_size);

    end
end
