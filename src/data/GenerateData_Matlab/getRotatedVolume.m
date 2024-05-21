function [rotVol, rotPts, rotOrig, mask] = getRotatedVolume(vol, pts, rotMat,invRotMat, orig)

% get min of volume as filler value
fillVal = min(vol(:));

rotPts = [];
rotOrig = [];

% get vertices
sz = size(vol);
for idx=0:7
   %vertArray(idx+1, :) = de2bi(idx, 3, 'left-msb') .* sz;
   vertArray(idx+1, :) = int2bit(idx, 3).' .* sz;
end
vertArray(find(vertArray == 0)) = 1; % from 0-indexing to 1-indexing
vertArray(:, [1, 2]) = vertArray(:, [2, 1]);

% convert to new coordinates
for idx = 1:8
    vertArray(idx, :) = (rotMat * (vertArray(idx, :) - orig).').';
end

% get limits
minVert = floor(min(vertArray));
maxVert = ceil(max(vertArray));
rotSz = maxVert - minVert + 1;
rotVol = fillVal * ones(rotSz);
mask = zeros(rotSz);

for row = minVert(1):maxVert(1)
    for col = minVert(2):maxVert(2)
        for slc = minVert(3):maxVert(3)

            % get indices in vol, Shawn: RHS&LHS? maybe use reverse rotMat (rotMatR = axang2rotm([ax, -randAng]);)
            volIdx = (invRotMat * [col, row, slc].').' + orig; % calculations in RHS
            volIdx = [volIdx(2), volIdx(1), volIdx(3)]; % convert to LHS
            volIdx = round(volIdx);

            % check limits
            if volIdx(1) > 0 && volIdx(2) > 0 && volIdx(3) > 0 && ...
               volIdx(1) <= sz(1) && volIdx(2) <= sz(2) && volIdx(3) <= sz(3)

                % assign intensity
                rotIdx = [row, col, slc] - minVert + 1;
                rotVol(rotIdx(1), rotIdx(2), rotIdx(3)) = vol(volIdx(1), volIdx(2), volIdx(3));
                mask(rotIdx(1), rotIdx(2), rotIdx(3)) = 1;
            end
        end
    end
end

if ~isempty(pts)

    %convert to RHS for point conversions
    minVert = [minVert(2), minVert(1), minVert(3)];

    % find where ctr ends up in the rotated image.
    rotOrig = -minVert + 1;

    % rotate points around the center of the original image
    rotPts = (rotMat * (pts - orig).').' - minVert + 1;
end
