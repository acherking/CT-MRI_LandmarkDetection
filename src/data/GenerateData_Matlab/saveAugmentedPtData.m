
function saveAugmentedPtData(vol, pts, nAug, patName)

% reduce size
imgSize = [176 176 48];
%cropScale = 3/4;

% in mm
% oriRes = [0.15, 0.15, 0.15];
oriRes = [0.2604, 0.2604, 0.2604];

minMax = 3096;

orig = mean(pts);

idx = 1;
while idx <= nAug
    % if patName = 'LG' and idx < 38
    %   continue
    % end

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
        augVolSize = size(augVol);
        % the narrow (doesn't include some border area) region where has sth from the patient
        [augMask] = prepareMaskE(augVol);
        % for CT Post, because of...
        vol(find(augVol > minMax)) = minMax;
        origBase = "/data/gpfs/projects/punim1836/Data/raw/aug/";
        origFile = origBase + 'original_augmentation/' + patName + '_aug_' + strIdx + '.mat';
        save(origFile, 'augVol', 'augPts', 'augMask', "augVolSize", '-v7.3');
        fprintf("Saved augmentation vol for patient: %s -- %d \n To Path: %s\n", patName, idx, origFile)
        
        [augVolRescaled, augPtsRescaled, augMaskRescaled] = rescaleData(augVol, augPts, imgSize, augMask);
        
        augVolRescaledSize = size(augVolRescaled);
        
        scale = augVolRescaledSize ./ augVolSize;
        res = oriRes ./ scale;

        strSize = num2str(imgSize(1)) + "x" + num2str(imgSize(2)) + "x" + num2str(imgSize(3));
        inPath = origBase + "/reduce_size/" + strSize + "/";
        augRescaledFile = inPath + patName + '_' + strSize + '_' + strIdx + '.mat';
                
        % save to datastores
        save(augRescaledFile, 'augVolRescaled', 'augPtsRescaled', 'res', ...
            'augVolSize', 'augMaskRescaled', '-v7.3');

        fprintf("Saved augmentation vol for patient: %s -- %d \n To Path: %s\n", patName, idx, augRescaledFile)

        
        idx = idx + 1;
    end
end