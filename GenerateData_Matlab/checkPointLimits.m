function res = checkPointLimits(sz, pts)

res = true;
for idx = 1:size(pts, 1)
   pt = pts(idx, :);
   if ~isempty(find(pt < 10)) || ~isempty(find(sz - pt < 9))
       res = false;
       break;
   end
end