
function output_valid=check_valid_net_output(output_info)

% output_valid=true;
if isempty(output_info)
    output_valid=false;
else
    output_valid=output_info.forward_finished;
%     if isfield(output_info, 'valid_data')
%         if ~output_info.valid_data
%             output_valid=false;
%         end
%     end
end

end