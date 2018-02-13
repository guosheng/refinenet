

% example code for evaluation of saved prediction masks, e.g., producing IoU scores

function result_info=demo_evaluate_saved_prediction_voc()

addpath('./my_utils');

% provide class info, here's an example for VOC dataset.
class_info=gen_class_info_voc();

% replace by your prediction mask dir:
predict_result_dir='../cache_data/test_examples_voc/result_20180212182355_evaonly_custom_data_valset_5scales/predict_result_fused/predict_result_mask';
% predict_result_dir='../cache_data/test_examples_voc/result_20180212182355_evaonly_custom_data_valset_5scales/predict_result_3/predict_result_mask';

% replace by your groundtruth mask dir:
gt_mask_dir='../datasets/voc2012_trainval/SegmentationClass';

result_evaluate_param=[];
result_evaluate_param.predict_result_dir=predict_result_dir;
result_evaluate_param.gt_mask_dir=gt_mask_dir;

result_cached_filename='eva_result_info.mat';
result_cached_dir=fileparts(result_evaluate_param.predict_result_dir);
result_cached_file=fullfile(result_cached_dir, result_cached_filename);

diary_dir=result_cached_dir;
mkdir_notexist(diary_dir);
diary(fullfile(diary_dir, 'output.txt'));
diary on

result_info=evaluate_predict_results(result_evaluate_param, class_info);
disp_evaluate_result(result_info, class_info);

fprintf('saving evaluation result to: %s\n', result_cached_file);
save(result_cached_file, 'result_info');

my_diary_flush();
diary off

end

