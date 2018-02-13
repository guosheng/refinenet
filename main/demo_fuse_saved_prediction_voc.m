
function demo_fuse_saved_prediction_voc()

addpath('./my_utils');

% privoding preidction mask folders, e.g., results generated using different scales. should replace by your folders here:
predict_result_dirs=[];
predict_result_dirs{end+1}='../cache_data/test_examples_voc/result_20180212182355_evaonly_custom_data_valset_5scales/predict_result_1';
predict_result_dirs{end+1}='../cache_data/test_examples_voc/result_20180212182355_evaonly_custom_data_valset_5scales/predict_result_2';
predict_result_dirs{end+1}='../cache_data/test_examples_voc/result_20180212182355_evaonly_custom_data_valset_5scales/predict_result_3';
predict_result_dirs{end+1}='../cache_data/test_examples_voc/result_20180212182355_evaonly_custom_data_valset_5scales/predict_result_4';
predict_result_dirs{end+1}='../cache_data/test_examples_voc/result_20180212182355_evaonly_custom_data_valset_5scales/predict_result_5';

% cache dir for fused results
fuse_result_dir='../cache_data/test_examples_voc/result_20180212182355_evaonly_custom_data_valset_5scales/predict_result_fused_mscale';

% provide class info, here's an example for VOC dataset, should be replace by your implementation
class_info=gen_class_info_voc();


diary_dir=fuse_result_dir;
mkdir_notexist(diary_dir);
diary(fullfile(diary_dir, 'output.txt'));
diary on


fprintf('\n\n--------------------------------------------------\n\n');
disp('fusing multiscale predictions');

fuse_param=[];
fuse_param.predict_result_dirs=predict_result_dirs;
fuse_param.fuse_result_dir=fuse_result_dir;
fuse_param.cache_fused_score_map=true;
fuse_multiscale_results(fuse_param, class_info);

fprintf('\n\n--------------------------------------------------\n\n');
disp('fused results are saved in:');
disp(fuse_param.fuse_result_dir);


my_diary_flush();
diary off

end


