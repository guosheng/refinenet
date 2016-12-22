

function my_epoch_model_cache(opts, epoch, exp_info, net_config)
        
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
        cnn_save_model(epoch_dir, net_config, exp_info);
    end
    
    
    if do_cache_snapshot && opts.do_cache_model
        
        rmdir(snapshot_dir_bak, 's');
        movefile(snapshot_dir, snapshot_dir_bak, 'f');
        mkdir_notexist(snapshot_dir);
        fprintf('update: %s\n', snapshot_dir_bak);
                        
        net_config.ref.cache_filename='net-config-snapshot.mat';
        exp_info.cache_filename='exp-info-snapshot.mat';
        cnn_save_model(snapshot_dir, net_config, exp_info);
        
    end
    
       
    
end


