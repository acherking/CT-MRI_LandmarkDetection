
function saveDividedAugmentedPtData(vol, pts, nAug, patName)

leftPts = pts(1:2, :);
leftPts(:, 1) = leftPts(:, 1) - 510;
rightPts = pts(3:4, :);
rightPts(:, 1) = 510 - rightPts(:, 1) + 1;

leftOrig = mean(leftPts);
rightOrig = mean(rightPts);

leftVol = vol(511:end, :, :);
rightVol = vol(1:510, :, :);

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
    
        
    origBase = "F:\Data\augmentation_exp/";

    origLeftFile = origBase + patName + '_augLeft_' + strIdx + '.mat';
    save(origLeftFile, 'augLeftVol', 'augLeftPts', '-v7.3');
    fprintf("Saved augmentation vol for patient: %s -- %d \n To Path: %s\n", patName, idx, origLeftFile)

    origRightFile = origBase + patName + '_augRight_' + strIdx + '.mat';
    save(origRightFile, 'augRightVol', 'augRightPts', '-v7.3');
    fprintf("Saved augmentation vol for patient: %s -- %d \n To Path: %s\n", patName, idx, origRightFile)

    % reduce size
    % [augVol, augPts] = rescaleData(augVol, augPts, imgSz);
    % strIdx = num2str(idx);
    % fileName_vol = [inPath patName '_17017030_AugVol_' strIdx '.mat'];
    % fileName_pts = [outPath patName '_17017030_AugPts_' strIdx '.mat'];
                
    % save to datastores
    % rescaled_aug_vol = augVol;
    % save(fileName_vol, 'rescaled_aug_vol', '-v7.3');
    % fprintf("Saved augmentation vol for patient: %s -- %d \n To Path: %s\n", patName, idx, fileName_vol)

    % rescaled_aug_pts = augPts(:);
    % save(fileName_pts, 'rescaled_aug_pts', '-v7.3');
    % fprintf("Saved augmentation pts for patient: %s -- %d \n To Path: %s\n\n", patName, idx, fileName_pts)
        
    idx = idx + 1;
end