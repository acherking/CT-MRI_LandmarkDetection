
imageDataPath = '/Volumes/Shawn_HDD/PhD/Project/Date/CT_MRI_Pre_Post/AZ MR';

% get volume data
vol = dicomreadVolume(imageDataPath);
vol = squeeze(vol);

%%

level = multithresh(vol);
seg_I = imquantize(vol,level, [0 ,1]);

%%
slice =80;
BW = seg_I(:, :, slice);

subplot(2,2,1);
imshow(vol(:, :, slice), []);
title('Original');

subplot(2,2,2);
imshow(BW);
title('Binary');

subplot(2,2,3);
CH = bwconvhull(BW);
imshow(CH);
title('Union Convex Hull');