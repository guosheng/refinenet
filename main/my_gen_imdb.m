

function imdb=my_gen_imdb(train_opts, ds_info)


task_info_train=[];
task_info_train.data_type='train';
task_info_train.task_idxes=ds_info.train_idxes;

task_info_val=[];
task_info_val.data_type='val';
task_info_val.task_idxes=ds_info.test_idxes;


imdb=[];
imdb.ds_info=ds_info;

imdb.task_info_train=task_info_train;
imdb.task_info_val=task_info_val;

task_idxes=[];
task_idxes=cat(1, task_idxes, task_info_train.task_idxes);
task_idxes=cat(1, task_idxes, task_info_val.task_idxes);

imdb.max_task_idx=max(task_idxes);
imdb.task_idxes=task_idxes;

make_ref_obj(imdb);

end

