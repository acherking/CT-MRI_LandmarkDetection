
% imageDataPath = '/Volumes/Shawn_HDD/PhD/Project/Date/CT_MRI_Pre_Post/AZ Pre';
imageDataPath = '/data/gpfs/projects/punim1836/Data/raw/CT_MRI_Pre_Post/SM Pre';
filePath = [imageDataPath, '/ser003img00001.dcm'];

% info = dicominfo(filePath);
% info.RescaleSlope
% info.RescaleIntercept
% get volume data
vol = dicomreadVolume(imageDataPath);
vol = squeeze(vol);
min(vol(:))
max(vol(:))

%%

level = multithresh(vol);
seg_I = imquantize(vol,3318, [0 ,1]);

%%
sizeValue = size(seg_I);
mask = zeros(sizeValue);
for s = 1:sizeValue(3)
    mask(:,:,s) = bwconvhull(seg_I(:, :, s));
end

%%
mask = uint8(mask);

%%
imageDataPathPre = '/data/gpfs/projects/punim1836/Data/raw/CT_MRI_Pre_Post/SM Pre';
imageDataPathPost = '/data/gpfs/projects/punim1836/Data/raw/CT_MRI_Pre_Post/SM Post';
volPre = dicomreadVolume(imageDataPathPre);
volPost = dicomreadVolume(imageDataPathPost);
volPre = squeeze(volPre);
volPost = squeeze(volPost);
%%
mask = prepareMaskE(volPost, volPre);

%%
slice =131;

subplot(2,2,1);
imshow(volPost(:, :, slice), []);
title('Original');

subplot(2,2,2);
imshow(seg_I(:, :, slice));
title('Binary');

subplot(2,2,3);
imshow(mask(:, :, slice), []);
title('Union Convex Hull');
%%
slice = 25;

subplot(2,1,1);
imshow(augVolRescaled(:, :, slice), []);
title('Original');

subplot(2,1,2);
imshow(augMaskRescaled(:, :, slice), []);
title('Union Convex Hull');
%%
ptNames = ["AH", 'AZ', 'DE', 'DM', 'DM2', 'DGL', 'FA', 'GE', 'GM', 'GP', 'HB', 'HH', 'JH', 'JM', 'LG', 'LP', 'MJ', 'NV', 'PH', 'SM'];

for patient=ptNames
    
end
