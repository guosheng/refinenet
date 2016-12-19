

% Author: Guosheng Lin (guosheng.lin@gmail.com)


function exp_info = my_net_tool(net_config, imdb, opts)


exp_info=[];
assert(isa(net_config, 'ref_obj'));

work_infos=cell(0);

work_info_trn=gen_work_info_basic();
work_info_trn.ref.task_info=imdb.ref.task_info_train;
work_info_trn.ref.run_type='train';
work_info_trn.ref.split_name='train';
work_infos{end+1, 1}=work_info_trn;


work_info_eva_val=gen_work_info_basic();
work_info_eva_val.ref.task_info=imdb.ref.task_info_val;
work_info_eva_val.ref.run_type='evaluate';
work_info_eva_val.ref.split_name='val';
work_infos{end+1, 1}=work_info_eva_val;


cnn_load_snapshot(opts.model_snapshot_dir, net_config);
init_exp_info(opts, work_info_trn, work_info_eva_val, net_config);

start_epoch=work_info_trn.ref.current_epoch+1;



if opts.do_train
    
    my_clear_net_trn(net_config);    
    
else
    
    opts.eva_epoch_idxes_trn=start_epoch;
    opts.model_cache_epoch_idxes_snapshot=start_epoch;
    opts.model_cache_epoch_idxes=inf;
end


%override some settings, after snapshot loading:
work_info_eva_val.ref.eva_epoch_idxes=opts.eva_epoch_idxes_val;


global_init_fn=net_config.ref.global_init_fn;
for w_idx=1:length(work_infos)
    
    one_work_info=work_infos{w_idx};
    one_work_info.ref.run_trn=strcmp(one_work_info.ref.run_type, 'train');
    one_work_info.ref.run_eva=strcmp(one_work_info.ref.run_type, 'evaluate');
    
    if ~isempty(global_init_fn)
        global_init_fn(opts, imdb, one_work_info, net_config);
    end
    
end



for epoch=start_epoch:opts.epoch_num

    if opts.do_train
        work_info_trn.ref.current_epoch=epoch;
        my_net_run_epoch(opts, imdb, work_info_trn, net_config);    
    end
    
    if opts.do_eva_val
        work_info_eva_val.ref.current_epoch=epoch;
        if any(work_info_eva_val.ref.eva_epoch_idxes==epoch)
            my_net_run_epoch(opts, imdb, work_info_eva_val, net_config);    
        end
    end
    
  
  exp_info=[];
  exp_info.train=work_info_trn;
  exp_info.eva_val=work_info_eva_val;
    
   
  if ~isempty(net_config.ref.net_progress_disp_fn)
      net_config.ref.net_progress_disp_fn(opts, exp_info, imdb, net_config);
  end
  
  
  do_save_result(opts, epoch, exp_info, net_config);
    
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






function do_save_result(opts, epoch, exp_info, net_config)
        
    if ~opts.do_cache_model
        return;
    end
        
    model_cache_epoch_idxes=opts.model_cache_epoch_idxes;
    model_cache_epoch_idxes_snapshot=opts.model_cache_epoch_idxes_snapshot;

    do_cache=false;
    if any(epoch==model_cache_epoch_idxes)
        do_cache=true;
    end
        
    do_cache_snapshot=false;    
    if any(epoch==model_cache_epoch_idxes_snapshot)
        do_cache_snapshot=true;
    end
    
    if ~do_cache_snapshot && ~do_cache
        return;
    end
    
    model_cache_dir=opts.model_cache_dir;
    epoch_dir=fullfile(model_cache_dir, ['epoch_' num2str(epoch)]);

    snapshot_dir=opts.model_snapshot_dir;
    snapshot_dir_bak=fullfile(model_cache_dir, 'bak_snapshot');
    
    if do_cache
        mkdir_notexist(epoch_dir);
    end
    
    if do_cache_snapshot
        mkdir_notexist(snapshot_dir);
        mkdir_notexist(snapshot_dir_bak);
    end
    
   
  
  exp_info.train=exp_info.train.ref;
  exp_info.eva_val=exp_info.eva_val.ref;
    
  % clear tmp_cache before saving
  exp_info.train.tmp_cache=[];
  exp_info.eva_val.tmp_cache=[];
  
    
    if do_cache && opts.do_cache_model
                
        net_config_file=sprintf('net-config-epoch-%d.mat',epoch);
        net_config.ref.cache_filename=net_config_file;

        info_file=sprintf('exp-info-epoch-%d.mat',epoch);
        exp_info.cache_filename=info_file;
        cnn_save_snapshot(epoch_dir, net_config, exp_info);
    end
    
    
    if do_cache_snapshot && opts.do_cache_model
        
        rmdir(snapshot_dir_bak, 's');
        movefile(snapshot_dir, snapshot_dir_bak, 'f');
        mkdir_notexist(snapshot_dir);
        fprintf('update: %s\n', snapshot_dir_bak);
                        
        net_config.ref.cache_filename='net-config-snapshot.mat';
        exp_info.cache_filename='exp-info-epoch-snapshot.mat';
        cnn_save_snapshot(snapshot_dir, net_config, exp_info);
        
    end
    
       
    
end







function init_exp_info(opts, work_info_trn, work_info_eva_val, net_config)

    if isfield(net_config.ref, 'init_exp_info') && ~isempty(net_config.ref.init_exp_info)
          
          tmp_info=net_config.ref.init_exp_info;
          update_work_info_from_cache(work_info_trn, tmp_info, 'train');
          update_work_info_from_cache(work_info_eva_val, tmp_info, 'eva_val');
          
          current_epoch1=work_info_trn.ref.current_epoch;
          current_epoch2=work_info_eva_val.ref.current_epoch;
          current_epoch=max(current_epoch1, current_epoch2);
          
          net_config.ref.init_exp_info=[];
          
    else
          if isfield(opts, 'net_start_epoch') && ~isempty(opts.net_start_epoch)
            current_epoch=opts.work_info_missed_start_epoch-1;
          else
            current_epoch=0;
          end
    end
  
  work_info_trn.ref.current_epoch=current_epoch;
  work_info_eva_val.ref.current_epoch=current_epoch;
   
  
  fprintf('init from snpashot, epoch:%d\n', current_epoch);
    
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




