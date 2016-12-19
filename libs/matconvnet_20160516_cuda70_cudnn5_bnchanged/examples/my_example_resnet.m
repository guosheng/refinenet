


run  ../matlab/vl_setupnn

% download a pre-trained CNN from the web (needed once)
% urlwrite(...
%   'http://www.vlfeat.org/matconvnet/models/imagenet-googlenet-dag.mat', ...
%   'imagenet-googlenet-dag.mat') ;

% model_path='/hdrive2/segmentation/model_trained/matconvnet/imagenet-resnet-50-dag.mat';
model_path='/hdrive2/segmentation/model_trained/matconvnet/imagenet-resnet-50-dag_nobnorm.mat';

% load the pre-trained CNN
net = dagnn.DagNN.loadobj(load(model_path)) ;
net.mode = 'test' ;

tmp_var_idx=net.getVarIndex('res5cx');
net.vars(tmp_var_idx).precious=true;



% load and preprocess an image
im = imread('peppers.png') ;
im_ = single(im) ; % note: 0-255 range
im_ = imresize(im_, net.meta.normalization.imageSize(1:2)) ;
im_ = bsxfun(@minus, im_, net.meta.normalization.averageImage) ;

% run the CNN
net.eval({'data', im_}) ;


tmp_v=net.vars(tmp_var_idx).value;
disp(max(tmp_v(:)));
disp(mean(tmp_v(:)));

% obtain the CNN otuput
scores = net.vars(net.getVarIndex('prob')).value ;
scores = squeeze(gather(scores)) ;

% show the classification results
[bestScore, best] = max(scores) ;
figure(1) ; clf ; imagesc(im) ;
title(sprintf('%s (%d), score %.3f',...
net.meta.classes.description{best}, best, bestScore)) ;