function [transVol, transPts] = getTranslatedVolume(vol, pts)

% get random translation
maxTrans = 5;
randTans = rand(1, 3);
trans = round(2 * maxTrans .* randTans - maxTrans);

% define translated volume
sz = size(vol);
transSz = sz + abs(trans);
transVol = min(vol(:)) * ones(transSz);

% get start and end positions in the new volume
% and point locations
transPts = [pts(:, 2), pts(:, 1), pts(:, 3)];
startPos = [1, 1, 1];
for idx = 1:3
   if trans(idx) > 0
       startPos(:, idx) = trans(:, idx) + 1;
       transPts(:, idx) = transPts(:, idx) + trans(idx);
   end
end
transPts = [transPts(:, 2), transPts(:, 1), transPts(:, 3)];
endPos = startPos + sz - 1;

% get translated volume
transVol(startPos(1):endPos(1), startPos(2):endPos(2), startPos(3):endPos(3)) = vol;
