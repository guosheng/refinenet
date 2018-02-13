



function result_info=my_gen_seg_result(seg_param, ds_info, predict_info)
    
    result_info=[];
    
    img_size=seg_param.img_size_input;
       
    gt_mask=seg_param.gt_mask_input;
    assert(~isempty(gt_mask));
    gt_mask=uint8(gt_mask);
    assert(all(img_size==size(gt_mask)));
        
    
    eva_param=seg_param.eva_param;
    class_info=eva_param.class_info;
            
    assert(~isempty(predict_info));
    
    score_map=single(predict_info.score_map);
    score_map_org=score_map;
    
    score_map_size=size(score_map);
    score_map_size=score_map_size(1:2);
        
    if any(img_size~=score_map_size)
        score_map=log(score_map);
        score_map=max(score_map, -20);
        score_map=my_resize(score_map, img_size);
        score_map=exp(score_map);
    end
       

    [~, predict_mask]=max(score_map,[],3);
    predict_mask=uint8(gather(predict_mask));
               
        
    one_seg_eva_result=seg_eva_one_img(predict_mask, gt_mask, class_info);
    result_info.seg_eva_result=one_seg_eva_result;
       
    coarse_gt_mask=imresize(gt_mask, score_map_size, 'nearest');
    [~, coarse_predict_mask]=max(score_map_org,[],3);
    coarse_predict_mask=uint8(gather(coarse_predict_mask));
    one_seg_eva_result_coarse=seg_eva_one_img(coarse_predict_mask, coarse_gt_mask, class_info);
    result_info.seg_eva_result_coarse=one_seg_eva_result_coarse;
        
    
    predict_result_densecrf=[];
    result_info.seg_eva_result_densecrf=[];
    
    if eva_param.eva_densecrf_postprocess
        task_config=[];
        task_config.img_data=seg_param.img_data_input;
        task_config.score_map=score_map;
        predict_result_densecrf=gen_prediction_densecrf(task_config, eva_param);
        one_seg_eva_result_densecrf=seg_eva_one_img(predict_result_densecrf.predict_mask, gt_mask, class_info);
        result_info.seg_eva_result_densecrf=one_seg_eva_result_densecrf;
    end
       
    
    do_save_results(ds_info, seg_param, predict_mask, score_map_org, predict_result_densecrf);
      
end



function do_save_results(ds_info, seg_param, predict_mask_net, score_map_org, predict_result_densecrf)

    predict_mask_net=gather(predict_mask_net);
    score_map_org=gather(score_map_org);

    img_idx=seg_param.img_idx;
    eva_param=seg_param.eva_param;
    class_info=eva_param.class_info;
    
    predict_mask_data=class_info.class_label_values(predict_mask_net);      
    assert(isa(class_info.class_label_values, 'uint8'));	
    assert(isa(predict_mask_data, 'uint8'));
    
    img_name=ds_info.img_names{img_idx};
    
    if eva_param.save_predict_mask
        tmp_dir=eva_param.predict_result_dir_mask;
        mkdir_notexist(tmp_dir);
        one_cache_file=fullfile(tmp_dir, [img_name '.png']);
        imwrite(predict_mask_data, class_info.mask_cmap, one_cache_file);
        
    end
    
    if eva_param.save_predict_result_full
            
        % notes: saved score map values range from 0 to 255
        score_map_org=im2uint8(score_map_org);
        tmp_dir=eva_param.predict_result_dir_full;
        one_cache_file=fullfile(tmp_dir, [img_name '.mat']);
        tmp_result_info=[];
        tmp_result_info.mask_data=predict_mask_data;
        tmp_result_info.score_map=score_map_org;
        tmp_result_info.img_size=size(predict_mask_data);
        tmp_result_info.class_info=class_info;
        my_save_file(one_cache_file, tmp_result_info, true, true);    		
    end
    
       
    
    if ~isempty(predict_result_densecrf)

    	assert(eva_param.predict_save_mask)
        tmp_dir=eva_param.predict_result_dir_densecrf;
        mkdir_notexist(tmp_dir);
        save_mask_data=class_info.class_label_values(predict_result_densecrf.predict_mask);      
		assert(isa(save_mask_data, 'uint8'));
        one_cache_file=fullfile(tmp_dir, [img_name '.png']);
        imwrite(save_mask_data, class_info.mask_cmap, one_cache_file);
   
    end
    
    
end
