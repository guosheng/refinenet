

classdef My_net_util
    
    
    

methods(Static)

function group_info=gen_group_info_basic(net_config, group_name)

net_config.ref.group_counter=net_config.ref.group_counter+1;


group_info=[];
group_info.child_group_idxes=[];
group_info.net_info=[];

group_info.child_relation='chain';
group_info.group_idx=net_config.ref.group_counter;
group_info.prediction_layer_idxes=[];
group_info.forward_evaluate_fn=[];

group_info.forward_begin_fn=[];
group_info.forward_finish_fn=[];
group_info.skip_forward=false;
group_info.skip_backward=false;
group_info.name=group_name;

group_info.use_dagnn=false;


end



function net_info=gen_net_info_basic()


net_info=[];
net_info.bp_start_layer=1;
net_info.do_bp=true;
net_info.bp_start_epoch=1;

net_info.net_stay_on_gpu=true;
net_info.data_stay_on_gpu=true;

net_info.lr_multiplier=1;
net_info.current_lr=0;

net_info.tmp_data=[];

net_info.layers=[];
net_info.name='';
net_info.net_on_gpu=false;

make_ref_obj(net_info);

end





function fix_padding_resnet(dag_net)


    for l_idx=1:length(dag_net.layers)
        l=dag_net.layers(l_idx);
        block=l.block;
       

        if isprop(block, 'pad')
                if isprop(block, 'poolSize')
                    filter_size=block.poolSize;
                else
                    filter_size=block.size(1:2);
                end
                assert(~isempty(filter_size));

%                 if isa(block, 'dagnn.Pooling')
%                     disp(block);
%                 end

                pad_size1=round((filter_size(1)-1)/2);
                pad_size2=round((filter_size(2)-1)/2);
                assert(pad_size2==pad_size1);

                pad_vs=unique(block.pad);
                if numel(pad_vs)>1
                    block.pad=[pad_size1, pad_size1, pad_size2, pad_size2];
%                     fprintf('fix padding for layer and block:\n');
%                     disp(l)
%                     disp(block);
                else
                    block.pad=pad_size1;
%                     assert(pad_vs==pad_size1);
                end
        end
    end
end


function fix_batch_norm_resnet(dag_net)

    for l_idx=1:length(dag_net.layers)
        l=dag_net.layers(l_idx);
        block=l.block;

        if isa(block, 'dagnn.BatchNorm')

           tmp_param_idx=l.paramIndexes(end);
           dag_net.params(tmp_param_idx).trainMethod='fixed';
           tmp_name=dag_net.params(tmp_param_idx).name;
           assert(~isempty(strfind(tmp_name, '_moments')));

        end
    end

end








function outputs=add_dropout_dagnn(dag_net, prev_layer_name, inputs)
    
name=[prev_layer_name '_dropout'];

outputs={[name '_varout']};

block = dagnn.DropOut() ;
% block.leak = net.layers{l}.leak ;


dag_net.addLayer(...
    name, ...
    block, ...
    inputs, ...
    outputs, ...
    {}) ;
      

end






function [outputs, one_output_dim]=add_joint_layer_dagnn_matchdim(dag_net, inputs, name, joint_input_dims)


outputs={[name '_varout']};

block = dagnn.Sum() ;

block.numInputs=2;

dag_net.addLayer(...
    name, ...
    block, ...
    inputs, ...
    outputs, ...
    {}) ;

assert(joint_input_dims(1)==joint_input_dims(2));
one_output_dim=joint_input_dims(1);

      
  
end




function [outputs, layer_name]=add_dim_reduce_layer(dag_net, input_var_name, input_dim, target_dim)


layer_name=[input_var_name '_dimred'];
[outputs, layer_name]=My_net_util.add_dim_reduce_layer_named(dag_net, input_var_name, input_dim, target_dim, layer_name);

end



function [outputs, layer_name]=add_dim_reduce_layer_named(dag_net, input_var_name, input_dim, target_dim, layer_name)

inputs={input_var_name};
outputs={[layer_name '_varout']};
filter_size=[3 3 input_dim, target_dim];
My_net_util.add_conv_layer_dagnn(dag_net, inputs, outputs, false, filter_size, layer_name, false)

end





function add_conv_layer_dagnn(dag_net, inputs, outputs, conv_t_flag, filter_size, name, hasBias)


    
    params = struct(...
        'name', {}, ...
        'value', {}, ...
        'learningRate', [], ...
        'weightDecay', []) ;
   
    
    
     filters= 0.01 * randn(filter_size(1),filter_size(2),filter_size(3), filter_size(4),'single');
    
     
     if conv_t_flag
         assert(~hasBias);
     end
        
      sz = size(filters) ;
      params(1).name = [name '_filter'] ;
      params(1).value = filters ;
      params(1).learningRate = 1;
      params(1).weightDecay = 1;
      if hasBias
      
          if conv_t_flag
              biases= zeros(1, filter_size(3), 'single');
%               biases= zeros(1, 1, 'single');
          else
              biases= zeros(1, filter_size(4), 'single');
          end
        
        params(2).name = [name '_bias'];
        params(2).value = biases ;
        params(2).learningRate = 1 ;
        params(2).weightDecay = 1;
        
      end
      
      if conv_t_flag
          block = dagnn.ConvTranspose() ;
          block.size = sz ;
          block.hasBias = hasBias ;
          block.upsample = [2 2];
          block.crop=0;
          block.numGroups=filter_size(4);
      else
          
          pad_size1=round((filter_size(1)-1)/2);
          
          block = dagnn.Conv() ;
          block.size = sz ;
          block.hasBias = hasBias ;
          block.pad = pad_size1;
          block.stride = 1;
      end
          

    dag_net.addLayer(...
    name, ...
    block, ...
    inputs, ...
    outputs, ...
    {params.name}) ;

  for p = 1:numel(params)
    pindex = dag_net.getParamIndex(params(p).name) ;
    if ~isempty(params(p).value)
      dag_net.params(pindex).value = params(p).value ;
    end
    if ~isempty(params(p).learningRate)
      dag_net.params(pindex).learningRate = params(p).learningRate ;
    end
    if ~isempty(params(p).weightDecay)
      dag_net.params(pindex).weightDecay = params(p).weightDecay ;
    end
  end
        
  
end


function outputs=add_relu_dagnn(dag_net, prev_layer_name, inputs)
    
name=[prev_layer_name '_relu'];

outputs={[name '_varout']};

block = dagnn.ReLU() ;
% block.leak = net.layers{l}.leak ;


dag_net.addLayer(...
    name, ...
    block, ...
    inputs, ...
    outputs, ...
    {}) ;
      

end






function [layer_gen_info, conv_layer_names, joint_layer_names]=...
    add_res_conv_block(dag_net, layer_gen_info, output_dim, conv_num, layer_name_prefix)

    if conv_num<=0
        return;
    end

    one_output_dim=layer_gen_info.one_output_dim;
    one_outputs=layer_gen_info.one_outputs;
    block_output_dim=output_dim;        
        
    assert(block_output_dim==one_output_dim);
        
    conv_layer_names=cell(conv_num, 1);                
    joint_layer_names=cell(conv_num, 1);
    
    for one_conv_idx=1:conv_num
                
        
        name_prefix=sprintf([layer_name_prefix '_b%d'], one_conv_idx);
        
        one_input_dim=one_output_dim;
        
        one_imap_start_outputs=one_outputs;
        one_imap_start_feat_dim=one_input_dim;
                
        one_outputs=My_net_util.add_relu_dagnn(dag_net, [name_prefix '_prev'], one_outputs);
        
        filter_size=[3 3 one_input_dim block_output_dim]; 
        one_inputs=one_outputs;
        filter_name=[name_prefix '_conv'];
        one_outputs={[filter_name '_outvar']};
        My_net_util.add_conv_layer_dagnn(dag_net, one_inputs, one_outputs, false, filter_size, filter_name, true);
        
        conv_layer_names{one_conv_idx}=filter_name;
        
        one_output_dim=block_output_dim;
                
        one_outputs=My_net_util.add_relu_dagnn(dag_net, filter_name, one_outputs);

        feat_dim_before=one_output_dim;
        feat_dim_after=one_output_dim;
        one_outputs=My_net_util.add_dim_reduce_layer(dag_net, one_outputs{1}, feat_dim_before, feat_dim_after);
        
                
        joint_layer_name=[name_prefix '_joint'];
        assert(length(one_imap_start_outputs)==1);
        one_inputs={one_imap_start_outputs{1} one_outputs{1}};
        joint_input_dims=[one_imap_start_feat_dim, one_output_dim];
        [one_outputs, joint_output_dim]=My_net_util.add_joint_layer_dagnn_matchdim(dag_net, one_inputs, joint_layer_name, joint_input_dims);
        
        start_output_var_idx=dag_net.getVarIndex(one_imap_start_outputs{1});
        dag_net.vars(start_output_var_idx).fanout=dag_net.vars(start_output_var_idx).fanout+1;
        
        one_output_dim=joint_output_dim;
        
    
        joint_layer_names{one_conv_idx}=joint_layer_name;
        
       
    end
        
        
    layer_gen_info.one_output_dim=one_output_dim;
    layer_gen_info.one_outputs=one_outputs;
        

end




function outputs=add_var_copy_dagnn(dag_net, prev_layer_name, inputs)
    
name=[prev_layer_name '_copy'];

outputs={[name '_varout']};

block = My_copy_layer ;

dag_net.addLayer(...
    name, ...
    block, ...
    inputs, ...
    outputs, ...
    {}) ;
     

end







end %end methods



end %end classdef


