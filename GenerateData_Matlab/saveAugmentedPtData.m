
function saveAugmentedPtData(vol, pts, nAug, inPath, outPath, imgIdx, nameStr)

imgSz = [170 170 30]; % change as required
%cropScale = 3/4;

idx = 1;
while idx <= nAug

    % save the original data first, then start the augmentation
    if idx == 1
        augVol = vol;
        augPts = pts; 
        orig = mean(pts);
    else
        [augVol, augPts] = getAugmentedVolume(vol, pts, orig); 
    end
    
    % crop to include only the middle region   
    %[augVol, augPts] = getCroppedVolume(augVol, augPts, cropScale);   
    
    if checkPointLimits(size(augVol), augPts)
        
        % reduce size
        
        [augVol, augPts] = rescaleData(augVol, augPts, imgSz);
        strIdx = (imgIdx - 1) * nAug + idx
        fileName = [nameStr{strIdx}, '.mat'];       
                
        % save to datastores
        v = augVol;
        save([inPath, fileName], 'v');

        v = augPts(:);
        save([outPath, fileName], 'v');
        
        idx = idx + 1;
    end
end