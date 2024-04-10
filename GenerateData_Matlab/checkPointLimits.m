function res = checkPointLimits(augSz, augPts)

res = true;
augPts(:, [1, 2]) = augPts(:, [2, 1])
for idx = 1:size(augPts, 1)
   pt = augPts(idx, :);
   if ~isempty(find(pt < 10)) || ~isempty(find(augSz - pt < 9))
       res = false;
       break;
   end
end