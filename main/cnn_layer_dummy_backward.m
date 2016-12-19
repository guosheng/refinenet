

function input_info=cnn_layer_dummy_backward(input_info, layer, work_info_batch, output_info)


input_info.dzdx=output_info.dzdx;


end