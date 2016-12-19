
function group_info=get_current_work_group_idx(work_info_batch)

group_idx=work_info_batch.ref.net_run_current_group_idx;
net_run_current_group_idx_linked=work_info_batch.ref.net_run_current_group_idx_linked;
if ~isempty(net_run_current_group_idx_linked)
    group_idx=net_run_current_group_idx_linked;
end
group_info=work_info_batch.ref.net_config.ref.group_infos{group_idx};

end