function [] = resize_volume(patName, volume, pts, newSize, idx, desDir)

% reduce size
[rescaled_aug_vol, rescaled_aug_pts] = rescaleData(volume, pts, newSize);

strIdx = num2str(idx);
strSize = num2str(newSize);
strSize = strSize(~isspace(strSize));
desFileName = [desDir patName '_' strSize '_VolPts_' strIdx '.mat'];
                
% save to datastores
save(desFileName, 'rescaled_aug_vol', 'rescaled_aug_pts', '-v7.3');
fprintf("Saved augmentation vol&pts for patient: %s -- %d \n To Path: %s\n", patName, idx, desFileName)
