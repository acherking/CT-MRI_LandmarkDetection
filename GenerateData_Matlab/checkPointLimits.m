function res = checkPointLimits(augSz, augPts)

res = true;
for idx = 1:size(augPts, 1)
   pt = augPts(idx, :);
   if ~isempty(find(pt < 10)) || ~isempty(find(augSz - pt < 9))
       res = false;
       break;
   end
end