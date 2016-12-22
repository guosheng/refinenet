

function net_config=gen_network_main(train_opts)


fprintf('\n generate network...\n\n');
    
net_config=gen_init_net_config();

net_config.ref.group_counter=0;

net_config.ref.root_group_info=My_net_util.gen_group_info_basic(net_config, 'root');

group_output_info=gen_multi_featnet_group(train_opts, net_config);

group_output_info=gen_network_refine_group(train_opts, net_config, group_output_info);

loss_group_info=gen_network_loss_group(train_opts, net_config, group_output_info);

net_config.ref.predict_group_idx=loss_group_info.group_idx;
net_config.ref.valid_group_flags=true(length(net_config.ref.group_infos), 1);
net_config.ref.root_group_idx=net_config.ref.root_group_info.group_idx;
net_config.ref.group_infos{net_config.ref.root_group_info.group_idx, 1}=net_config.ref.root_group_info;

net_config.ref.net_input_img_scales=train_opts.net_input_img_scales;

gen_dagnn_info(net_config);
    
end



function net_config=gen_init_net_config()

net_config=[];
net_config.group_counter=0;
net_config.group_infos=cell(0);
net_config.root_group_idx=[];
net_config.valid_group_flags=[];

make_ref_obj(net_config);

end



function group_output_info=gen_multi_featnet_group(train_opts, net_config)


input_featnet_num=length(train_opts.input_featnet_configs);

child_output_infos=cell(input_featnet_num, 1);

multi_featnet_group_info=My_net_util.gen_group_info_basic(net_config, 'multi_featnet');
multi_featnet_group_info.child_relation='parallel';


for featnet_idx=1:input_featnet_num
           
    featnet_config=train_opts.input_featnet_configs{featnet_idx};
    [featnet_group_info, one_net_output_info]=gen_featnet_group_info(train_opts, net_config, featnet_config);
    multi_featnet_group_info.child_group_idxes(end+1,1)=featnet_group_info.group_idx;
    child_output_infos{featnet_idx}=one_net_output_info;
end

net_config.ref.group_infos{multi_featnet_group_info.group_idx,1}=multi_featnet_group_info;
net_config.ref.root_group_info.child_group_idxes(end+1,1)=multi_featnet_group_info.group_idx;

group_output_info=[];
group_output_info.child_infos=child_output_infos;

end






function gen_dagnn_info(net_config)


group_num=length(net_config.ref.group_infos);
dag_group_flags=false(group_num,1);


for g_idx=1:group_num

    one_group_info=net_config.ref.group_infos{g_idx};
    
    if isfield(one_group_info, 'use_dagnn') && one_group_info.use_dagnn
        dag_group_flags(g_idx)=true;
    end
end

net_config.ref.dag_group_flags=dag_group_flags;

   

end



function [featnet_group_info, net_output_info]=gen_featnet_group_info(train_opts, net_config, featnet_config)
          
    dag_net=[];
    featnet_type=featnet_config.featnet_type;
    
    net_output_info=[];

    if strcmp(featnet_type, 'resnet')
        [layers, net_output_info, dag_net]=gen_network_featnet_resnet(featnet_config);
    end

    if strcmp(featnet_type, 'vgg')
        error('not support!')
    end

    if strcmp(featnet_type, 'imgraw')
        [layers, net_output_info, dag_net]=gen_network_featnet_imgraw(featnet_config);
    end
        
    
    assert(~isempty(net_output_info));
    
    net_info=My_net_util.gen_net_info_basic();
    net_info.ref.name=sprintf('featnet_%d', featnet_config.featnet_idx);
    net_info.ref.layers=layers;
    net_info.ref.lr_multiplier=featnet_config.lr_multiplier;
    
        
    % control bp here:
    net_info.ref.do_bp=train_opts.input_net_do_bp;

    % or control the layer for bp:
    % net_info.ref.bp_start_layer=length(layers);

    % if init from a trained model, init will be performed up to this group
%     net_info.ref.stop_network_init=true;
        
    featnet_group_info=My_net_util.gen_group_info_basic(net_config, net_info.ref.name);
    featnet_group_info.net_info=net_info;
    featnet_group_info.dag_net=dag_net;
    if ~isempty(dag_net)
        featnet_group_info.use_dagnn=true;
    end
    net_config.ref.group_infos{featnet_group_info.group_idx, 1}=featnet_group_info;
    
end




