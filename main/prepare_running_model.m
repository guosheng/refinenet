

function [net_config, net_exp_info]=prepare_running_model(train_opts)

if train_opts.run_evaonly
    
    [net_config, net_exp_info]=do_prepare_evaonly(train_opts);
    
else
    
    [net_config, net_exp_info]=do_prepare_train(train_opts);
    
end

end


function [net_config, net_exp_info]=do_prepare_evaonly(train_opts)


fprintf('\n\n-------------------------------------------------------------\n\n');
disp('prepare_running_model_evaonly...');

assert(~isempty(train_opts.trained_model_path));
    
fprintf('load trained model:\n');
disp(train_opts.trained_model_path);
net_config=cnn_load_model(train_opts.trained_model_path);
if isempty(net_config)
    error('trained_model is not existed!');
end
          
net_exp_info=[];

fprintf('\n\n-------------------------------------------------------------\n\n');

end



function [net_config, net_exp_info]=do_prepare_train(train_opts)


fprintf('\n\n-------------------------------------------------------------\n\n');
disp('prepare_running_model...');


[net_config_input, net_exp_info]=cnn_load_model(train_opts.model_snapshot_dir);
if ~isempty(net_config_input)
    fprintf('model cached snapshot existed, resume training:\n');
    disp(train_opts.model_snapshot_dir);
    assert(~isempty(net_exp_info));
else
    net_exp_info=[];
    fprintf('model snapshot not existed\n');
end


if isempty(net_config_input)
    
    if ~isempty(train_opts.trained_model_path)
        fprintf('load trained model:\n');
        disp(train_opts.trained_model_path);
        net_config_input=cnn_load_model(train_opts.trained_model_path);
    %     assert(~isempty(net_config_input));
        if isempty(net_config_input)
            error('trained_model is not existed!');
        end
    end
    
else
    
    if ~isempty(train_opts.trained_model_path)
        fprintf('WARNING: the specified trained model is not used!! using the cached model snapshot instead!\n');
    end
end


net_config=train_opts.gen_network_fn(train_opts);
if ~isempty(net_config_input)
    fprintf('init model from cache...\n');
    
    % should not directly use net_config_input for training.
        
    % here we init the layers in the new net_config using the layers the cached model.
    % this is to avoid using the parameter setting saved
    % in the cached model, and ensure the settings in train_opts make effects.
    
    init_model_from_cache(train_opts, net_config, net_config_input);
end
           

fprintf('\n\n-------------------------------------------------------------\n\n');

end


