
function my_diary_flush()

tmp_status=get(0,'Diary');
if strcmp(tmp_status, 'on')
    diary off
    diary on
end

end