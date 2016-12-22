

% Author: Guosheng Lin (guosheng.lin@gmail.com)


function exp_info = my_net_tool(opts, imdb, net_config, input_exp_info)

net_run_opts=opts.net_run_opts;

exp_info=[];
assert(isa(net_config, 'ref_obj'));

work_infos=cell(0);

work_info_trn=gen_work_info_basic();
work_info_trn.ref.run_type='train';
work_info_trn.ref.split_name='train';
work_infos{end+1, 1}=work_info_trn;

work_info_eva_val=gen_work_info_basic();
work_info_eva_val.ref.run_type='evaluate';
work_info_eva_val.ref.split_name='val';
work_infos{end+1, 1}=work_info_eva_val;

for w_idx=1:length(work_infos)
    one_work_info=work_infos{w_idx};
    one_work_info.ref.run_trn=strcmp(one_work_info.ref.run_type, 'train');
    one_work_info.ref.run_eva=strcmp(one_work_info.ref.run_type, 'evaluate');
end

do_init_exp_info(opts, work_info_trn, work_info_eva_val, input_exp_info);
clear input_exp_info

work_info_trn.ref.task_info=imdb.ref.task_info_train;
work_info_eva_val.ref.task_info=imdb.ref.task_info_val;


start_epoch=work_info_trn.ref.current_epoch+1;

if opts.do_train
    
    my_clear_net_trn(net_config);    
    
end



global_init_fn=net_run_opts.global_init_fn;
if ~isempty(global_init_fn)
    if opts.do_train
        global_init_fn(opts, imdb, work_info_trn, net_config);
    end
    
    if opts.do_eva_val
        global_init_fn(opts, imdb, work_info_eva_val, net_config);
    end
end



for epoch=start_epoch:opts.epoch_num

    if opts.do_train
        work_info_trn.ref.current_epoch=epoch;
        my_net_run_epoch(opts, imdb, work_info_trn, net_config);    
    end
    
    if opts.do_eva_val
        work_info_eva_val.ref.current_epoch=epoch;
        if any(opts.eva_epoch_idxes==epoch)
            my_net_run_epoch(opts, imdb, work_info_eva_val, net_config);    
        end
    end
    
  
  exp_info=[];
  exp_info.train=work_info_trn;
  exp_info.eva_val=work_info_eva_val;
    
   
  if ~isempty(net_run_opts.net_progress_disp_fn)
      net_run_opts.net_progress_disp_fn(opts, exp_info, imdb, net_config);
  end
    
  opts.epoch_model_cache_fn(opts, epoch, exp_info, net_config);
  
    
  my_diary_flush();
  
  if opts.do_train
      if work_info_trn.ref.epoch_valid_batch_count==0
          fprintf('\n\n===work_info_trn.ref.epoch_valid_batch_count==0, stop \n\n');
          break;
      end
  end
        
end


my_diary_flush();


end




function work_info=gen_work_info_basic()

work_info=[];
work_info.objective = [] ;
work_info.eva_results = cell(0, 1) ;
work_info.current_epoch=0;
work_info.valid_epoch_idxes=[];
work_info.total_iter_num=0;
work_info.tmp_cache=[];


make_ref_obj(work_info);

end



function do_init_exp_info(opts, work_info_trn, work_info_eva_val, input_exp_info)

    if ~isempty(input_exp_info)
                    
          update_work_info_from_cache(work_info_trn, input_exp_info, 'train');
          update_work_info_from_cache(work_info_eva_val, input_exp_info, 'eva_val');
          
          current_epoch1=work_info_trn.ref.current_epoch;
          current_epoch2=work_info_eva_val.ref.current_epoch;
          current_epoch=max(current_epoch1, current_epoch2);
                    
    else
          if isfield(opts, 'net_start_epoch') && ~isempty(opts.net_start_epoch)
            current_epoch=opts.work_info_missed_start_epoch-1;
          else
            current_epoch=0;
          end
    end
  
  work_info_trn.ref.current_epoch=current_epoch;
  work_info_eva_val.ref.current_epoch=current_epoch;
   
    
end




function update_work_info_from_cache(work_info, tmp_info, work_info_name)

if isfield(tmp_info, work_info_name)
    
      cached_work_info=tmp_info.(work_info_name);
      
      if isa(cached_work_info, 'ref_obj')
          cached_work_info=cached_work_info.ref;
      end
            
      work_info.ref=cached_work_info;
      work_info.ref.tmp_cache=[];
      
end
  

end




