

function my_epoch_run_config(opts, imdb, work_info, net_config, work_info_epoch)


do_config_basic(opts, imdb, work_info, net_config, work_info_epoch);

      
run_trn=work_info.ref.run_trn;
if run_trn
    do_config_train(opts, imdb, work_info, net_config, work_info_epoch);
end


run_eva=work_info.ref.run_eva;
if run_eva
    do_config_eva(opts, imdb, work_info, net_config, work_info_epoch);
end


torun_task_subidxes=find(work_info_epoch.ref.valid_task_flags);
torun_task_idxes=work_info.ref.task_info.task_idxes(torun_task_subidxes);
assert(~isempty(torun_task_idxes));


if run_trn
    assert(~run_eva)
    assert(all(ismember(torun_task_idxes, imdb.ref.ds_info.train_idxes)))

    shuffle_idxes=randperm(length(torun_task_subidxes));
    torun_task_subidxes = torun_task_subidxes(shuffle_idxes) ;
end


if run_eva
    assert(~run_trn)
    assert(all(ismember(torun_task_idxes, imdb.ref.ds_info.test_idxes)))
end


work_info_epoch.ref.batch_torun_task_subidxes=torun_task_subidxes;
work_info_epoch.ref.task_run_count=0;


end



function do_config_basic(opts, imdb, work_info, net_config, work_info_epoch)
    

    input_task_num=length(work_info.ref.task_info.task_idxes);
       
    work_info_epoch.ref.task_finish_flags=false(input_task_num, 1);
    work_info_epoch.ref.task_finish_progresses=zeros(input_task_num, 1);
    work_info_epoch.ref.input_task_num=input_task_num;
    
    work_info_epoch.ref.valid_task_flags=true(input_task_num, 1);
    work_info_epoch.ref.batch_size=opts.batch_size;
    
    work_info_epoch.ref.eva_result=[];
    work_info_epoch.ref.done_task_num=0;
    work_info_epoch.ref.task_num=input_task_num;
            
            
end  





function do_config_train(opts, imdb, work_info, net_config, work_info_epoch)
       
    if ~opts.epoch_task_subsample
        return;
    end
    
    input_task_num=work_info_epoch.ref.input_task_num;
        
    if opts.epoch_task_num_min>=input_task_num
        return;
    end
                
    epoch_run_priority_task_flags=[];
    if isfield(work_info.ref, 'epoch_run_priority_task_flags')
       epoch_run_priority_task_flags=work_info.ref.epoch_run_priority_task_flags;
    end

    if isfield(work_info.ref, 'epoch_run_count_imgs')
        epoch_run_count_imgs=work_info.ref.epoch_run_count_imgs;
    else
        epoch_run_count_imgs=zeros(input_task_num, 1, 'uint32');
    end

    if isempty(epoch_run_priority_task_flags)
        epoch_run_priority_task_flags=true(input_task_num, 1);
    end


    can_task_subidxes=find(epoch_run_priority_task_flags);
    can_task_subidxes=my_randperm(can_task_subidxes);
    can_task_subidxes2=find(~epoch_run_priority_task_flags);
    if ~isempty(can_task_subidxes2)
        can_task_subidxes2=my_randperm(can_task_subidxes2);
        can_task_subidxes=cat(1, can_task_subidxes, can_task_subidxes2);
    end


    valid_task_flags=gen_epoch_task_info(opts, imdb, work_info, work_info_epoch, can_task_subidxes);


    epoch_run_priority_task_flags(valid_task_flags)=false;
    if ~any(epoch_run_priority_task_flags)
        epoch_run_priority_task_flags=[];
    end
    work_info.ref.epoch_run_priority_task_flags=epoch_run_priority_task_flags;


    epoch_run_count_imgs(valid_task_flags)=epoch_run_count_imgs(valid_task_flags)+1;
    work_info.ref.epoch_run_count_imgs=epoch_run_count_imgs;

    assert(~isempty(valid_task_flags));
    work_info_epoch.ref.valid_task_flags=valid_task_flags;

end


function valid_task_flags=gen_epoch_task_info(opts, imdb, work_info, work_info_epoch, can_task_subidxes)

input_task_num=work_info_epoch.ref.input_task_num;
if(input_task_num<=opts.epoch_task_num_min)
    valid_task_flags=true(input_task_num, 1);
    return;
end

valid_task_flags=[];

if strcmp(opts.epoch_task_gen_type_train, 'class_sample')
    
    valid_task_flags=do_gen_epoch_tasks_class_sample(opts, imdb, work_info, work_info_epoch, can_task_subidxes);
    
    valid_num=nnz(valid_task_flags);
    if valid_num<opts.epoch_task_num_min
        non_valid_flags=~valid_task_flags;
        tmp_task_subidxes=can_task_subidxes(non_valid_flags(can_task_subidxes));
        need_sample=opts.epoch_task_num_min-valid_num;
        tmp_task_subidxes=tmp_task_subidxes(1:need_sample);
        valid_task_flags(tmp_task_subidxes)=true;
    end
    
    if valid_num>opts.epoch_task_num_max
        tmp_task_sub_idxes=find(valid_task_flags);
        tmp_task_sub_idxes=my_randperm(tmp_task_sub_idxes);
        tmp_task_sub_idxes=tmp_task_sub_idxes(1:opts.epoch_task_num_max);
        valid_task_flags=false(input_task_num, 1);
        valid_task_flags(tmp_task_sub_idxes)=true;
    end
    
end

if strcmp(opts.epoch_task_gen_type_train, 'random')
    valid_task_flags=false(input_task_num, 1);
    tmp_task_sub_idxes=can_task_subidxes;
    if length(tmp_task_sub_idxes)>opts.epoch_task_num_max
        tmp_task_sub_idxes=tmp_task_sub_idxes(1:opts.epoch_task_num_max);
    end
    valid_task_flags(tmp_task_sub_idxes)=true;
end

assert(~isempty(valid_task_flags));


end



function valid_task_flags=do_gen_epoch_tasks_class_sample(opts, imdb, work_info, work_info_epoch, can_task_subidxes)

    input_task_num=work_info_epoch.ref.input_task_num;
    epoch_run_max_task_one_class=opts.epoch_run_max_task_one_class;
    
    task_idxes=work_info.ref.task_info.task_idxes;

    class_info=imdb.ref.ds_info.class_info;
    class_num=class_info.class_num;
    train_exclude_class_idxes=class_info.void_class_idxes;
    class_idxes_imgs=imdb.ref.ds_info.class_idxes_imgs;

    img_count_classes=zeros(class_num, 1);
    img_count_classes(train_exclude_class_idxes)=inf;

    % indexed by task_subidx:
    valid_task_flags=false(input_task_num, 1);
    for can_idx=1:length(can_task_subidxes)
        one_task_subidx=can_task_subidxes(can_idx);
        one_task_idx=task_idxes(one_task_subidx);

        one_class_idxes=class_idxes_imgs{one_task_idx};

        current_exist_counts=img_count_classes(one_class_idxes);
        if any(current_exist_counts<epoch_run_max_task_one_class)
            valid_task_flags(one_task_subidx)=true;
            img_count_classes(one_class_idxes)=img_count_classes(one_class_idxes)+1;

            if min(img_count_classes)>=epoch_run_max_task_one_class
                break;
            end
        end
    end
    
    
       
end








function do_config_eva(opts, imdb, work_info, net_config, work_info_epoch)


if ~opts.eva_param.skip_existing_prediction_eva
    return;
end

input_task_num=work_info_epoch.ref.input_task_num;
predict_result_dir=opts.eva_param.predict_result_dir_mask;

task_idxes=work_info.ref.task_info.task_idxes;
valid_task_flags=false(input_task_num, 1);


skip_count=0;
ds_info=imdb.ref.ds_info;

for one_task_subidx=1:input_task_num
    
    img_idx=task_idxes(one_task_subidx);
    
    img_name=ds_info.img_names{img_idx};
        one_cache_file=fullfile(predict_result_dir, [img_name '.png']);
    
    if ~my_check_file(one_cache_file, true)
        one_cache_file=fullfile(predict_result_dir,...
            ['img_idx_' num2str(img_idx) '.png']);
    end
        
        
    if ~my_check_file(one_cache_file, true)
        valid_task_flags(one_task_subidx)=true;
    else
    	skip_count=skip_count+1;
    	if mod(skip_count,100)==1
        	fprintf('skip existed result: %s\n', one_cache_file);
    	end
    end

end

fprintf('skip existed result, skip count: %d\n', skip_count);

work_info_epoch.ref.valid_task_flags=valid_task_flags;

end




