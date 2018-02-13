



function eva_param=update_eva_param_mscale(eva_param, run_config) 


	eva_param.predict_result_dir_mask=fullfile(run_config.root_cache_dir, 'predict_result_mask');
    eva_param.predict_result_dir_full=fullfile(run_config.root_cache_dir, 'predict_result_full');
	

end
