


function label_counts=my_label_count(label_values, label_data)

    tmp_counts=accumarray(label_data, ones(size(label_data)), [max(label_values) 1]);
    label_counts=tmp_counts(label_values);

end