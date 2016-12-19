function input_info=my_init_input_info(input_info)

if input_info.is_group_data
   input_info_child_groups=input_info.data_child_groups;
   child_num=length(input_info_child_groups);
   for c_idx=1:child_num
       input_info_child_groups{c_idx}=my_init_input_info(input_info_child_groups{c_idx});
   end
   input_info.data_child_groups=input_info_child_groups;
   input_info.child_valid_flags=true(child_num, 1);
end

input_info.bp_finished=false;
input_info.forward_finished=true;

input_info.aux=[];
input_info.dzdx=[];
input_info.dzdw=[];


end