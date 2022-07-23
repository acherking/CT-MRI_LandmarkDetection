function [augVol, augPts] = getAugmentedVolume(vol, pts, orig)

maxAng = 15 * pi / 180;
randAng = 2 * rand * maxAng - maxAng;
ax = rand(1, 3);
rotMat = axang2rotm([ax, randAng]);

% get rotated volume and points
[augVol, augPts] = getRotatedVolume(vol, pts, rotMat, orig);

% get translated volume and points
[augVol, augPts] = getTranslatedVolume(augVol, augPts);

% % rescale augVol to be the same size as vol
% [augVol, augPts] = rescaleData(augVol, augPts, size(vol));
% 
% % crop to half size to make it easier on the training model
% [augVol, augPts] = getCroppedVolume(augVol, augPts, 0.5);
% 
% % make the image smaller still
% [augVol, augPts] = rescaleData(augVol, augPts, round(size(augVol)/2));



