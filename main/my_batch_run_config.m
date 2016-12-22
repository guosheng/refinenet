
function my_batch_run_config(opts, imdb, work_info, net_config, work_info_epoch, work_info_batch)

    do_init_work_info_batch(opts, imdb, work_info, net_config, work_info_epoch, work_info_batch)
    
    work_info_batch.ref.batch_skip=false;
    batch_size=work_info_epoch.ref.batch_size;
    
    torun_task_subidxes=work_info_epoch.ref.batch_torun_task_subidxes;
    task_run_count=work_info_epoch.ref.task_run_count;
    tmp_max_idx=length(torun_task_subidxes);
    tmp_start_idx=task_run_count+1;
    tmp_stop_idx=task_run_count+batch_size;
    tmp_stop_idx=min(tmp_stop_idx, tmp_max_idx);
    
    if tmp_start_idx>tmp_max_idx
        task_subidxes_batch=[];
    else
        task_subidxes_batch=torun_task_subidxes(tmp_start_idx:tmp_stop_idx);
        assert(~isempty(task_subidxes_batch));
    end
    
    task_run_count=tmp_stop_idx;
    work_info_epoch.ref.task_run_count=task_run_count;
    
    
        
    epoch_finished=false;
    if isempty(task_subidxes_batch)
        epoch_finished=true;
    end
        
    if ~epoch_finished
                        
        assert(size(task_subidxes_batch, 2)==1);

        batch_task_num=length(task_subidxes_batch);
        done_task_num=nnz(work_info_epoch.ref.task_finish_flags);
        input_task_num=work_info_epoch.ref.input_task_num;
        
        valid_task_num=input_task_num;
        valid_task_flags=work_info_epoch.ref.valid_task_flags;
        if ~isempty(valid_task_flags)
            valid_task_num=nnz(valid_task_flags);
        end
        
        work_info_epoch.ref.batch_task_num=batch_task_num;
        work_info_epoch.ref.done_task_num=done_task_num;
        work_info_epoch.ref.done_task_progress=done_task_num/valid_task_num;
                
        work_info_epoch.ref.task_finish_flags(task_subidxes_batch)=true;
               
        task_idxes_org=work_info.ref.task_info.task_idxes(task_subidxes_batch);
        work_info_batch.ref.task_idxes=task_idxes_org;
        work_info_batch.ref.task_subidxes=task_subidxes_batch;
        work_info_batch.ref.batch_task_num=batch_task_num;
                               
        
        ds_info= opts.get_batch_ds_info_fn(opts, imdb, work_info, ...
            net_config, work_info_epoch, work_info_batch) ;
        work_info_batch.ref.ds_info=ds_info;
        
        work_info_batch.ref.leaf_group_bp_flags=false(size(net_config.ref.group_infos));
                    
    end
    
    
    work_info_batch.ref.epoch_finished=epoch_finished;
                
end




function do_init_work_info_batch(opts, imdb, work_info, net_config, work_info_epoch, work_info_batch)


work_info_batch.ref.train_opts=opts;
work_info_batch.ref.imdb=imdb;
work_info_batch.ref.work_info=work_info;
work_info_batch.ref.work_info_epoch=work_info_epoch;
work_info_batch.ref.net_config=net_config;


work_info_batch.ref.run_trn=work_info.ref.run_trn;
work_info_batch.ref.run_eva=work_info.ref.run_eva;

work_info_batch.ref.epoch=work_info_epoch.ref.epoch;
work_info_batch.ref.total_iter_num=work_info.ref.total_iter_num;

work_info_batch.ref.net_run_verbose=opts.net_run_verbose;

if work_info.ref.run_trn
    work_info_batch.ref.gen_optimizer_param_fn=@my_gen_optimizer_param;
end

end



function optimizer_param=my_gen_optimizer_param(work_info_batch, net_info)

% this will be called in each batch...

optimizer_param=net_info.ref.tmp_data.optimizer_param;

end






