function [mask] = prepareMask(vol)
% Otsu's method
level = multithresh(vol);
seg_I = imquantize(vol,level, [0 ,1]);

% create mask
sizeValue = size(seg_I);
mask = zeros(sizeValue);
for s = 1:sizeValue(3)
    mask(:,:,s) = bwconvhull(seg_I(:, :, s));
end
% reduce store space
mask = uint8(mask);