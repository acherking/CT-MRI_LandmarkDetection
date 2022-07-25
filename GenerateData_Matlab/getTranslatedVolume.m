function [transVol, transPts] = getTranslatedVolume(rotVol, rotPts)

% get random translation, Shawn: what is translation used for?
maxTrans = 5;
randTans = rand(1, 3);
trans = round(2 * maxTrans .* randTans - maxTrans);

% define translated rotVolume
rotSz = size(rotVol);
transSz = rotSz + abs(trans);
transVol = min(rotVol(:)) * ones(transSz);

% get start and end positions in the new rotVolume
% and point locations
transPts = [rotPts(:, 2), rotPts(:, 1), rotPts(:, 3)];
startPos = [1, 1, 1];
for idx = 1:3
   if trans(idx) > 0
       startPos(:, idx) = trans(:, idx) + 1;
       transPts(:, idx) = transPts(:, idx) + trans(idx);
   end
end
transPts = [transPts(:, 2), transPts(:, 1), transPts(:, 3)];
endPos = startPos + rotSz - 1;

% get translated rotVolume
transVol(startPos(1):endPos(1), startPos(2):endPos(2), startPos(3):endPos(3)) = rotVol;
