

function do_bp_current_net=check_do_bp_current_net(work_info_batch, net_run_config, net_info)
    
    % check whether need to do bp:
    
    do_bp_current_net=net_run_config.do_bp;
    if ~isempty(net_info)
        do_bp_current_net=do_bp_current_net && net_info.ref.do_bp;
        if do_bp_current_net && net_info.ref.bp_start_epoch > work_info_batch.ref.epoch
            do_bp_current_net=false;
        end
    end
    
end