
function saveAugmentedPtData(vol, pts, nAug, inPath, outPath, patName)

imgSz = [170 170 30]; % change as required
%cropScale = 3/4;

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
    end
    
    % crop to include only the middle region   
    %[augVol, augPts] = getCroppedVolume(augVol, augPts, cropScale);   
    
    if checkPointLimits(size(augVol), augPts)
        
        origBase = "/Volumes/Shawn_HDD/PhD/Project/Date/augmentation_from_matlab/re_aug/original_augmentation_data/";
        origFile = origBase + patName + '_aug_' + strIdx + '.mat';
        save(origFile, 'augVol', 'augPts', '-v7.3');
        fprintf("Saved augmentation vol for patient: %s -- %d \n To Path: %s\n", patName, idx, origFile)

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
end