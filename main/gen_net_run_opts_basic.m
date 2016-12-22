


function net_run_opts=gen_net_run_opts_basic()


net_run_opts=[];
net_run_opts.global_init_fn=[];
net_run_opts.epoch_init_fn=[];
net_run_opts.epoch_finish_fn=[];
net_run_opts.batch_init_fn=[];
net_run_opts.batch_finish_fn=[];
net_run_opts.batch_info_disp_fn=[];
net_run_opts.epoch_info_disp_fn=[];
net_run_opts.batch_evaluate_fn=[];
net_run_opts.epoch_evaluate_fn=[];
net_run_opts.epoch_run_config_fn=[];
net_run_opts.batch_run_config_fn=[];
net_run_opts.net_progress_disp_fn=[];

net_run_opts.bp_start_epoch=1;
net_run_opts.net_run_verbose=false;


end