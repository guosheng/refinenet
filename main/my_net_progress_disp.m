

function my_net_progress_disp(opts, exp_info, imdb, net_config)

	if mod(exp_info.train.ref.current_epoch, opts.fig_plot_step)~=0
		return
	end


    if ~isfield(exp_info.train.ref.tmp_cache, 'epoch_eva_fig')
        f1=figure;
        exp_info.train.ref.tmp_cache.epoch_eva_fig=f1;
%         set(f1,'Visible', 'off'); 
    end
    epoch_eva_fig=exp_info.train.ref.tmp_cache.epoch_eva_fig;
%     figure(epoch_eva_fig);
    set(0,'CurrentFigure',epoch_eva_fig);
    
    
    

    plot_infos=cell(0, 1);
    one_plot_info=gen_plot_info(exp_info.train, 'train');
    if one_plot_info.valid_info
        one_plot_info.color='k';
        plot_infos{end+1}=one_plot_info;
    end
    one_plot_info=gen_plot_info(exp_info.eva_val, 'eva-val');
    if one_plot_info.valid_info
        one_plot_info.color='b';
        plot_infos{end+1}=one_plot_info;
    end
    
    if isempty(plot_infos)
        return;
    end

    tmp_plot_info=plot_infos{end};
    tmp_epoch=tmp_plot_info.valid_epoch_idxes(end);
    tmp_eva_result=tmp_plot_info.eva_results{tmp_epoch};
    assert(~isempty(tmp_eva_result));

    
        
    eva_names=tmp_eva_result.eva_names;
    eva_names_disp=tmp_eva_result.eva_names_disp;
    eva_name_num=length(eva_names);
    
    sub_plot_row_num=2;
    sub_plot_row_num=min(sub_plot_row_num, eva_name_num);
    
    sub_plot_clolmn_num=ceil(eva_name_num/sub_plot_row_num);


    modelFigPath = fullfile(opts.root_cache_dir, 'net_evaluate.pdf') ;
    modelFigPath_fig = fullfile(opts.root_cache_dir, 'net_evaluate.fig') ;

    
    clf ;
    subplot_counter=0;
 
  
      
  for eva_idx=1:eva_name_num
      
      eva_name=eva_names{eva_idx};
      eva_name_disp=eva_names_disp{eva_idx};
      
      subplot_counter=subplot_counter+1;
      subplot(sub_plot_row_num, sub_plot_clolmn_num, subplot_counter) ;
      one_legend_str=cell(0,1);
      for p_idx=1:length(plot_infos)
      
          plot_info=plot_infos{p_idx};
          valid_epoch_idxes=plot_info.valid_epoch_idxes;
          valid_ep_num=length(valid_epoch_idxes);
          eva_values=zeros(valid_ep_num,1);
          valid_ep_sel=false(valid_ep_num, 1);
          for tmp_ep_idx=1:valid_ep_num
              ep_idx=valid_epoch_idxes(tmp_ep_idx);
              eva_result=plot_info.eva_results{ep_idx};
              if isfield(eva_result, eva_name)
                  one_value=eva_result.(eva_name);
                  if ~isempty(one_value)
                      try
                      eva_values(tmp_ep_idx)=one_value;
                      valid_ep_sel(tmp_ep_idx)=true;
                      catch
                          keyboard;
                      end
                  end
              end
          end
          
          if nnz(valid_ep_sel)>0
              plot_values=eva_values(valid_ep_sel);
              valid_epoch_idxes=valid_epoch_idxes(valid_ep_sel);

              plot(valid_epoch_idxes, plot_values, plot_info.color) ; hold on ;
              one_legend_str{end+1}=plot_info.plot_name_disp;
          end
      end
                  
      xlabel('epoch') ; ylabel(eva_name_disp) ; 
      if ~isempty(one_legend_str)
        h=legend(one_legend_str) ; 
      end
      grid on ;
      set(h,'color','none') ;
      title(eva_name_disp) ;
      hold off;
  end
    
  
  
%   if exp_info.train.ref.current_epoch==1
%   % maximize
%         pause(0.1);
%         frame_h = get(handle(gcf),'JavaFrame');
%         set(frame_h,'Maximized',1);
%   end
  
  drawnow ;
  pause(0.5);
  print(1, modelFigPath, '-dpdf') ;
  saveas(gcf, modelFigPath_fig);
  
  fprintf('save figure: %s\n', modelFigPath);
  fprintf('save figure: %s\n', modelFigPath_fig);
    
  
end
    
  


function plot_info=gen_plot_info(work_info, plot_name_disp)
    
    plot_info=[];
    plot_info.eva_results=work_info.ref.eva_results;
    plot_info.valid_info=true;
    plot_info.plot_name_disp=plot_name_disp;
    
    if isempty(work_info.ref.eva_results)
        plot_info.valid_info=false;
        return;
    end
        
    valid_epoch_idxes=work_info.ref.valid_epoch_idxes;
    valid_epoch_idxes=valid_epoch_idxes(valid_epoch_idxes<=work_info.ref.current_epoch);
    plot_info.valid_epoch_idxes=valid_epoch_idxes;
        

end











