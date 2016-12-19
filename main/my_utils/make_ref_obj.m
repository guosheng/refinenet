


function make_ref_obj(input_var)

if isa(input_var, 'ref_obj')
    return;
end

one_mutable=ref_obj;
one_mutable.ref=input_var;
assignin('caller', inputname(1), one_mutable);

end