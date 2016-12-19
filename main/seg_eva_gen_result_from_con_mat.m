

function one_result=seg_eva_gen_result_from_con_mat(one_con_mat, exclude_class_idxes)

        class_num=size(one_con_mat, 1);

        accuracy_classes=zeros(class_num, 1);
        inter_union_score_classes=zeros(class_num, 1);
        
        if nargin<2
            exclude_class_idxes=[];
        end
        
        valid_gt_class_sel=true(class_num, 1);
     
        global_pos_num=0;
        global_predict_num=0;
        
        for c_idx=1:class_num
                       
            one_true_pos_num=one_con_mat(c_idx,c_idx);
            one_gt_pos_num=sum(one_con_mat(c_idx,:));
            one_predict_pos_num=sum(one_con_mat(:, c_idx));
            
            global_pos_num=global_pos_num+one_true_pos_num;
            global_predict_num=global_predict_num+one_predict_pos_num;
            
            if one_gt_pos_num>0
                one_accuracy=one_true_pos_num/(one_gt_pos_num+eps);
                one_inter_union_score=one_true_pos_num/...
                    (one_gt_pos_num+one_predict_pos_num-one_true_pos_num+eps);
                
                accuracy_classes(c_idx)=one_accuracy;
                inter_union_score_classes(c_idx)=one_inter_union_score;
            else
                valid_gt_class_sel(c_idx)=false;
            end
            
        end
        
        
        valid_class_sel=true(class_num, 1);
        valid_class_sel(exclude_class_idxes)=false;
        valid_class_sel=valid_class_sel & valid_gt_class_sel;
        
        accuracy_per_class=mean(accuracy_classes(valid_class_sel));
        inter_union_score_per_class=mean(inter_union_score_classes(valid_class_sel));
        accuracy_global=global_pos_num/global_predict_num;
                
        one_result.confusion_mat=one_con_mat;        
        one_result.class_num=class_num;
        
        one_result.accuracy_global=accuracy_global;
        one_result.accuracy_per_class=accuracy_per_class;
        one_result.inter_union_score_per_class=inter_union_score_per_class;
        one_result.accuracy_classes=accuracy_classes;
        one_result.inter_union_score_classes=inter_union_score_classes;
                        
        one_result.valid_class_num=nnz(valid_class_sel);
        one_result.valid_class_idxes=find(valid_class_sel);
        one_result.exclude_class_idxes=exclude_class_idxes;

end


