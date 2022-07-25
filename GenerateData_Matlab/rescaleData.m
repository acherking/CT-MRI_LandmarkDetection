function [augVol, pts, mask] = rescaleData(augVol, augPts, imgSz, mask)

% rescale augVol to be imgSz
augVolSz = size(augVol);
augVol = imresize3(augVol, imgSz);

% resample points
sc = (imgSz - 1) ./ (augVolSz - 1);
sc = [sc(2), sc(1), sc(3)];

for idx = 1:3
   pts(:, idx) = sc(idx) * (pts(:, idx) - 1) + 1; 
end

if nargin > 3
    mask = imresize3(mask, imgSz);
else
    mask = [];
end