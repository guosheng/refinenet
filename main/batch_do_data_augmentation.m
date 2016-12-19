

function batch_data=batch_do_data_augmentation(train_opts, imdb, work_info_epoch, batch_data, work_info_batch, work_info)


data_aug_config=train_opts.data_aug_config;
if work_info.ref.run_eva || ~data_aug_config.do_aug
    batch_data.aug_info=[];
    return;
end


if ~isfield(imdb.ref, 'data_aug_params') || isempty(imdb.ref.data_aug_params)
    imdb.ref.data_aug_params=gen_data_aug_params(train_opts);
end

aug_num=length(imdb.ref.data_aug_params);
aug_idx=randsample(aug_num, 1);

aug_param=imdb.ref.data_aug_params{aug_idx};
assert(~isempty(aug_param));
batch_data=apply_data_aug(batch_data, aug_param);

aug_info=[];
aug_info.aug_param=aug_param;
batch_data.aug_info=aug_info;


end


function data_aug_params=gen_data_aug_params(train_opts)

data_aug_params=cell(0);

data_aug_config=train_opts.data_aug_config;
if ~isempty(data_aug_config) && data_aug_config.do_aug
    config_counter=0;
    
    aug_scales=data_aug_config.aug_scales;
    if isempty(aug_scales)
        aug_scales=1;
    end
    
    aug_flips=data_aug_config.aug_flips;
    if isempty(aug_flips)
        aug_flips=false;
    end
    
    for s_idx=1:length(aug_scales)
        one_aug_scale=aug_scales(s_idx);
        for f_idx=1:length(aug_flips)
            one_aug_flip=aug_flips(f_idx);
            
            config_counter=config_counter+1;
            
            data_aug_param=[];
            data_aug_param.scale=one_aug_scale;
            data_aug_param.flip=one_aug_flip;
            data_aug_param.aug_idx=config_counter;
            data_aug_param.do_aug=true;
                        
            data_aug_params{config_counter,1}=data_aug_param;
        end
    end

    assert(config_counter>1);
end

end




function batch_data=apply_data_aug(batch_data, aug_param)

one_scale=aug_param.scale;
if one_scale~=1
    expect_size=round(size(batch_data.img_data).*one_scale);
    expect_size=expect_size(1:2);
    
    % make the size as an even number
    need_changed_flags=mod(expect_size, 2)~=0;
    expect_size(need_changed_flags)=expect_size(need_changed_flags)+1;
    
    batch_data.img_data=imresize(batch_data.img_data, expect_size);
    
end

if aug_param.flip
    batch_data.img_data = flip(batch_data.img_data,2); 
    assert(~isempty(batch_data.label_data));
    batch_data.label_data=flip(batch_data.label_data, 2);
end

end


