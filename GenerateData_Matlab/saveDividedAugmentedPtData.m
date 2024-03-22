
function saveDividedAugmentedPtData(vol, pts, nAug, patName)

imgSize = [176 88 48; 240 120 64; 320 160 96].';

volSize = size(vol);
halfXsize = volSize(2) / 2;

if volSize(2) ~= 1020
    fprintf("X not 1020, patient: %s (%d)\n", patName, volSize(2));
end

leftPts = pts(1:2, :);
leftPts(:, 1) = leftPts(:, 1) - halfXsize;
rightPts = pts(3:4, :);
rightPts(:, 1) = halfXsize - rightPts(:, 1) + 1;

leftOrig = mean(leftPts);
rightOrig = mean(rightPts);

leftVol = vol(:, (halfXsize+1):end, :);
rightVol = vol(:, 1:halfXsize, :);
rightVol = flip(rightVol, 2);

idx = 1;
while idx <= nAug

    strIdx = string(idx);
    fprintf("Start augmentation for patient: %s -- %d\n", patName, idx)

    % save the original data first, then start the augmentation
    if idx == 1
        augLeftVol = leftVol;
        augLeftPts = leftPts; 
        augRightVol = rightVol;
        augRightPts = rightPts;
    else
        [augLeftVol, augLeftPts] = getAugmentedVolume(leftVol, leftPts, leftOrig);
        while not (checkPointLimits(size(augLeftVol), augLeftPts))
            [augLeftVol, augLeftPts] = getAugmentedVolume(leftVol, leftPts, leftOrig);
        end

        [augRightVol, augRightPts] = getAugmentedVolume(rightVol, rightPts, rightOrig);
        while not (checkPointLimits(size(augLeftVol), augLeftPts))
            [augRightVol, augRightPts] = getAugmentedVolume(rightVol, rightPts, rightOrig);
        end
    end

    augLeftVolSize = size(augLeftVol);
    augRightVolSize = size(augRightVol);
        
    origBase = "F:\Data\augmentation_exp";

    augFile = origBase + "\original_divided_augmentation\" + patName + '_aug_' + strIdx + '.mat';
    %save(augFile, 'augLeftVol', 'augLeftPts', 'augLeftVolSize', ...
    %    'augRightVol', 'augRightPts', 'augRightVolSize', '-v7.3');

    %fprintf("Saved augmentation vol for patient: %s -- %d \n To Path: %s\n", patName, idx, augFile)

    % reduce size
    for s = imgSize
        sizeT = s.';
        [augLeftVolRescaled, augLeftPtsRescaled] = rescaleData(augLeftVol, augLeftPts, sizeT);
        [augRightVolRescaled, augRightPtsRescaled] = rescaleData(augRightVol, augRightPts, sizeT);
        
        augLeftVolRescaledSize = size(augLeftVolRescaled);
        augRightVolRescaledSize = size(augRightVolRescaled);

        % in mm
        oriRes = [0.15, 0.15, 0.15];
        
        leftScale = augLeftVolRescaledSize ./ augLeftVolSize;
        leftRes = oriRes ./ leftScale;

        rightScale = augRightVolRescaledSize ./ augRightVolSize;
        rightRes = oriRes ./ rightScale;

        strSize = num2str(sizeT(1)) + "x" + num2str(sizeT(2)) + "x" + num2str(sizeT(3));
        inPath = origBase + "\reduce_size\" + strSize + "\";
        augRescaledFile = inPath + patName + '_' + strSize + '_' + strIdx + '.mat';
                
        % save to datastores
        save(augRescaledFile, 'augLeftVolRescaled', 'augLeftPtsRescaled', 'leftRes', 'augLeftVolSize', ...
            'augRightVolRescaled', 'augRightPtsRescaled', 'rightRes', 'augRightVolSize','-v7.3');

        fprintf("Saved augmentation vol for patient: %s -- %d \n To Path: %s\n", patName, idx, augRescaledFile)

    end

    idx = idx + 1;
end