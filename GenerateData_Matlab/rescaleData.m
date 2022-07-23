function [vol, pts, mask] = rescaleData(vol, pts, sz, mask)

% rescale vol to be sz
volSz = size(vol);
vol = imresize3(vol, sz);

% resample points
sc = (sz - 1) ./ (volSz - 1);
sc = [sc(2), sc(1), sc(3)];

for idx = 1:3
   pts(:, idx) = sc(idx) * (pts(:, idx) - 1) + 1; 
end

if nargin > 3
    mask = imresize3(mask, sz);
else
    mask = [];
end