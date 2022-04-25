% i = 4;
% in_fn_test = fullfile(image_root_dir, strcat(dblm.rel_filenames{i}, '.jpg'));
% in_img = imread(in_fn_test);
img = "../images/test16.jpg"
in_img = imread(img);
load(['models/m170306_dlib2_xm2.mat']);
load(['../test.mat']);
addpath(genpath('code'));

% imshow(in_img)
B = reshape(arr',1,numel(arr));
lm = B;
% lm = dblm.data(i,:)+1;
[out_img, res] = regfunc_fanc_do_normalization(in_img, lm, model);

figure(1);

subplot(131);
% imshow(in_img); hold on; plot(lm(1:2:end), lm(2:2:end), 'g.'); hold off;
imshow(in_img);
title(img);

subplot(132);
imshow(in_img); hold on; plot(lm(1:2:end), lm(2:2:end), 'g.'); hold off;
title('image with landmarks');

subplot(133); 
imshow(out_img);
title('Result of frontalization');