function [mask] = prepareMaskE(volPost)
% Otsu's method
level = multithresh(volPost);
seg_I = imquantize(volPost,level, [0 ,1]);

% create mask
sizeValue = size(seg_I);
mask = zeros(sizeValue);
for s = 1:sizeValue(3)
    mask(:,:,s) = bwconvhull(seg_I(:, :, s));
end
%%
% exclude the electrodes
% maxVal = max(volPre(:));
seg_E = imquantize(volPost,3092, [1 ,0]);
mask = mask & seg_E;

% reduce store space
mask = uint8(mask);