
function saveDividedAugmentedPtData(vol, pts, nAug, patName)

imgSize = [176 88 48].';

% volSize = size(vol);

% if volSize(2) ~= 1020
%     fprintf("X not 1020, patient: %s (%d)\n", patName, volSize(2));
% end

orig = mean(pts);

idx = 1;
while idx <= nAug

    strIdx = string(idx);
    fprintf("Start augmentation for patient: %s -- %d\n", patName, idx)
    
    % save the original data first, then start the augmentation
    if idx == 1
        augVol = vol;
        augPts = pts; 
    else
        [augVol, augPts] = getAugmentedVolume(vol, pts, orig);
        while not (checkPointLimits(size(augVol), augPts))
            [augVol, augPts] = getAugmentedVolume(vol, pts, orig);
        end
    end

    augVolSize = size(augVol);
    halfXsize = round(augVolSize(2) / 2);

    augLeftPts = augPts(1:2, :);
    augLeftPts(:, 1) = augLeftPts(:, 1) - halfXsize;
    augRightPts = augPts(3:4, :);
    augRightPts(:, 1) = halfXsize - augRightPts(:, 1) + 1;

    augLeftVol = augVol(:, (halfXsize+1):end, :);
    augRightVol = augVol(:, 1:halfXsize, :);
    augRightVol = flip(augRightVol, 2);

    augLeftVolSize = size(augLeftVol);
    augRightVolSize = size(augRightVol);
        
    origBase = "/data/gpfs/projects/punim1836/Data/raw/aug";

    augFile = origBase + "/original_divided_augmentation/" + patName + '_aug_' + strIdx + '.mat';
    save(augFile, 'augVol', 'augPts', 'augVolSize', '-v7.3');

    fprintf("Saved augmentation vol for patient: %s -- %d \n To Path: %s\n", patName, idx, augFile)

    % reduce size
    for s = imgSize
        sizeT = s.';
        [augLeftVolRescaled, augLeftPtsRescaled] = rescaleData(augLeftVol, augLeftPts, sizeT);
        [augRightVolRescaled, augRightPtsRescaled] = rescaleData(augRightVol, augRightPts, sizeT);
        
        augLeftVolRescaledSize = size(augLeftVolRescaled);
        augRightVolRescaledSize = size(augRightVolRescaled);

        % in mm
        %oriRes = [0.15, 0.15, 0.15];
        oriRes = [0.26, 0.26, 0.3];
        
        leftScale = augLeftVolRescaledSize ./ augLeftVolSize;
        leftRes = oriRes ./ leftScale;

        rightScale = augRightVolRescaledSize ./ augRightVolSize;
        rightRes = oriRes ./ rightScale;

        strSize = num2str(sizeT(1)) + "x" + num2str(sizeT(2)) + "x" + num2str(sizeT(3));
        inPath = origBase + "/reduce_size/" + strSize + "/";
        augRescaledFile = inPath + patName + '_' + strSize + '_' + strIdx + '.mat';
                
        % save to datastores
        save(augRescaledFile, 'augLeftVolRescaled', 'augLeftPtsRescaled', 'leftRes', 'augLeftVolSize', ...
            'augRightVolRescaled', 'augRightPtsRescaled', 'rightRes', 'augRightVolSize','-v7.3');

        fprintf("Saved augmentation vol for patient: %s -- %d \n To Path: %s\n", patName, idx, augRescaledFile)

    end

    idx = idx + 1;
end