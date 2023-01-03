function [] = appendPixelDistance(patName, orgSize, newSize, idx, desDir)

% in mm
oriRes = [0.15, 0.15, 0.15];

scale = newSize ./ orgSize;
pixel_distance = oriRes ./ scale;

strIdx = num2str(idx);
strSize = num2str(newSize);
strSize = strSize(~isspace(strSize));
desFileName = [desDir patName '_' strSize '_VolPts_' strIdx '.mat'];
                
% save to datastores
save(desFileName, 'pixel_distance', '-append');
fprintf("Appended pixel distance (%.2d-%d-%d mm) for patient: %s -- %d \n To Path: %s\n",pixel_distance(1), pixel_distance(2), pixel_distance(3), patName, idx, desFileName)