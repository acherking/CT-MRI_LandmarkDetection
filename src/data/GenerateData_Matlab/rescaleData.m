function [rsVol, augPts, mask] = rescaleData(augVol, augPts, imgSz, mask)

% rescale augVol to be imgSz
idx_0 = augVol(find(augVol<0)); % for test
idx_1 = augVol(find(augVol>0)); % for test
idx_2 = augVol(find(augVol>1)); % for test
augVolSz = size(augVol);
rsVol = imresize3(augVol, imgSz);
idx_greater_0 = rsVol(find(rsVol>0)); % for test
idx_smaller_0 = find(rsVol<0); % for test
idx_greater_1 = rsVol(find(rsVol>1)); % for test

% resample points
sc = (imgSz - 1) ./ (augVolSz - 1);
sc = [sc(2), sc(1), sc(3)];

for idx = 1:3
   augPts(:, idx) = sc(idx) * (augPts(:, idx) - 1) + 1; 
end

if nargin > 3
    mask = imresize3(mask, imgSz);
else
    mask = [];
end