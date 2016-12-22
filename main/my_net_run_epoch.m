


function my_net_run_epoch(opts, imdb, work_info, net_config)

epoch_time = tic ;

net_run_opts=opts.net_run_opts;

run_type=work_info.ref.run_type;

epoch=work_info.ref.current_epoch;
work_info.ref.obj_values(epoch) = 0 ;
work_info.ref.eva_results{epoch}=[];


work_info_epoch=[];
work_info_epoch.epoch=epoch;
work_info_epoch.valid_batch_count=0;
work_info_epoch.gpu_reset_count=0;
make_ref_obj(work_info_epoch);


epoch_config_time=tic;
net_run_opts.epoch_run_config_fn(opts, imdb, work_info, ...
        net_config, work_info_epoch);
epoch_config_time=toc(epoch_config_time);
work_info_epoch.ref.epoch_config_time=epoch_config_time;


do_run_batch(opts, imdb, work_info, net_config, work_info_epoch);


valid_batch_count=work_info_epoch.ref.valid_batch_count;
if valid_batch_count>0
    net_run_opts.epoch_evaluate_fn(work_info, opts, imdb, net_config, work_info_epoch);
end

eva_result=work_info_epoch.ref.eva_result;
work_info.ref.eva_results{epoch}=eva_result;
work_info.ref.obj_values(epoch)=eva_result.obj_value;
work_info.ref.epoch_valid_batch_count=work_info_epoch.ref.valid_batch_count;

work_info.ref.valid_epoch_idxes=cat(1, work_info.ref.valid_epoch_idxes, epoch);

epoch_time=toc(epoch_time);
fprintf('\n==epoch_info, epoch:%d, epoch_time:%.1f, config_time:%.1f, obj:%.6f, run_type:%s, split:%s\n', ...
  epoch, epoch_time, epoch_config_time, ...
  eva_result.obj_value, run_type, work_info.ref.split_name) ;


if work_info.ref.run_trn
    if length(work_info.ref.obj_values)>1
      if work_info.ref.obj_values(epoch)>100*work_info.ref.obj_values(epoch-1)
          fprintf('\n\n\n### WARNING: the obj_values is jumping up quickly!\n\n\n');
    %           keyboard;
      end
    end
end


if ~isempty(net_run_opts.epoch_info_disp_fn)
    net_run_opts.epoch_info_disp_fn(opts, imdb, work_info, net_config, work_info_epoch);
end


my_diary_flush();
  



end









function do_run_batch(opts, imdb, work_info, net_config, work_info_epoch)

  net_run_opts=opts.net_run_opts;

  if ~isempty(net_run_opts.epoch_init_fn)
      net_run_opts.epoch_init_fn(opts, imdb, work_info, ...
            net_config, work_info_epoch);
  end
   
  epoch=work_info_epoch.ref.epoch;
      
  task_count=0;
  batch_idx=0;
  valid_batch_count=0;
        
  batch_run_config_fn=net_run_opts.batch_run_config_fn;
  batch_disp_thresh=0;
 

  
  while true
    
    batch_time = tic ;
    
    work_info.ref.total_iter_num=work_info.ref.total_iter_num+1;
    
    batch_idx=batch_idx+1;
    
    
    work_info_batch=[];
    work_info_batch.batch_idx=batch_idx;
    make_ref_obj(work_info_batch);
    
    batch_config_time=tic;
    batch_run_config_fn(opts, imdb, work_info, ...
        net_config, work_info_epoch, work_info_batch);
    batch_config_time=toc(batch_config_time);
    work_info_batch.ref.batch_config_time=batch_config_time;
    
    if work_info_batch.ref.batch_skip
        continue;
    end
        
    if work_info_batch.ref.epoch_finished
        break;
    end
    
    if ~isempty(net_run_opts.batch_init_fn)
        net_run_opts.batch_init_fn(opts, imdb, work_info, ...
            net_config, work_info_epoch, work_info_batch);
    end
    
    
    task_num_batch=work_info_batch.ref.batch_task_num;
       
    
    
    net_run_config=[];
    net_run_config.sync=true;
    net_run_config.do_bp=false;
    net_run_config.use_gpu=opts.use_gpu;
       
    
    work_info_batch.ref.run_backward=false;
    if work_info.ref.run_trn
        if epoch>=net_run_opts.bp_start_epoch
            net_run_config.do_bp=true;
            work_info_batch.ref.run_backward=true;            
        end
    end
    
    work_info_batch.ref.net_run_config=net_run_config;
      
    work_info_batch.ref.forward_time=0;
    work_info_batch.ref.backward_time=0;
    work_info_batch.ref.eva_time=0;
    
    net_time=tic;
    my_net_run_batch(net_config, work_info_batch);
    net_time=toc(net_time);
    
    one_forward_time=work_info_batch.ref.forward_time;    
    one_backward_time=work_info_batch.ref.backward_time;
       
      
      
    if ~isempty(net_run_opts.batch_finish_fn)  
        net_run_opts.batch_finish_fn(opts, imdb, work_info, ...
            net_config, work_info_epoch, work_info_batch);
    end
    
            
    % clear data_info
    work_info_batch.ref.data_info_groups=[];
    
        
    eva_time=tic;
    if ~isempty(net_run_opts.batch_evaluate_fn)
        net_run_opts.batch_evaluate_fn(...
            opts, imdb, net_config, work_info, work_info_epoch, work_info_batch);
    end
    eva_time=toc(eva_time);
    
    eva_time=eva_time+work_info_batch.ref.eva_time;
        
    
    valid_batch_count=valid_batch_count+1;
    task_count=task_count+task_num_batch;
    batch_time = toc(batch_time) ;
       
    
    task_finish_progress=work_info_epoch.ref.done_task_progress;
    
    if task_finish_progress>=batch_disp_thresh
        
        if opts.batch_disp_step<0
            batch_disp_thresh=0;
        else
            net_disp_count=floor(task_finish_progress./opts.batch_disp_step)+1;
            batch_disp_thresh=net_disp_count.*opts.batch_disp_step;
        end
        
        obj_value=work_info_epoch.ref.eva_result.obj_value;
                
        done_task_num=work_info_epoch.ref.done_task_num;
        task_num=work_info_epoch.ref.task_num;
        
        valid_task_num=task_num;
        valid_task_flags=work_info_epoch.ref.valid_task_flags;
        if ~isempty(valid_task_flags)
            valid_task_num=nnz(valid_task_flags);
        end
        
                
        fprintf('\n--batch_info, iter:%d, epoch:%d(total:%d), finished_task:%d, epoch_task:%d, total_task:%d, run_type:%s, split:%s\n', ...
            work_info.ref.total_iter_num, ...
            epoch, opts.epoch_num, ...
            done_task_num, valid_task_num, task_num, ...
            work_info.ref.run_type, work_info.ref.split_name) ;
        
        fprintf('--batch_info, gpu:%d, time:%.1f (forw:%.1f, backw:%.1f, net:%.1f, eva:%.1f), data_load_time:%.1f, obj:%.5f\n', ...
            opts.use_gpu, batch_time, one_forward_time, one_backward_time, net_time, eva_time, ...
            batch_config_time, obj_value) ;
            
        
       
       
        if work_info.ref.run_trn
            group_idxes_train=find(work_info_batch.ref.leaf_group_bp_flags);
            lr_groups=zeros(length(group_idxes_train),1);
            for tmp_n_idx=1:length(group_idxes_train)
                one_group_idx=group_idxes_train(tmp_n_idx);
                group_info=net_config.ref.group_infos{one_group_idx};
                lr_groups(tmp_n_idx)=group_info.net_info.ref.current_lr;
            end
            group_idxes_str=my_gen_array_str(group_idxes_train);
            lr_str=my_gen_array_str(lr_groups, '%.0e');

            fprintf('--batch_info, group:%s, learn_rate:%s\n', group_idxes_str, lr_str) ;
        end
        
        if ~isempty(net_run_opts.batch_info_disp_fn)
            net_run_opts.batch_info_disp_fn(opts, work_info_batch);
        end
        
    end
       
                    
    clear work_info_batch
           
  end
  
  
  work_info_epoch.ref.task_count=task_count;
  work_info_epoch.ref.valid_batch_count=valid_batch_count;
  
  
  if ~isempty(net_run_opts.epoch_finish_fn)
      net_run_opts.epoch_finish_fn(opts, imdb, work_info, ...
            net_config, work_info_epoch);
  end
  
end






