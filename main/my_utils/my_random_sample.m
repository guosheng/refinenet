
function b_set=my_random_sample(a_set, sample_num)

a_num=length(a_set);

if a_num>sample_num
    b_set=a_set(randsample(a_num, sample_num));
else
    b_set=a_set;
end


end