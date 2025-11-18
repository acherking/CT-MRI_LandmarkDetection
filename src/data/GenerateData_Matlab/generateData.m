clear all;
close all;

% reset random value generator
rng default

roiFile = '/data/gpfs/projects/punim1836/CT-MRI_LandmarkDetection/data/processed/Y/more_CT_Pre/ROI_addition_CT_Pre_14_Nov2025.xlsx'; % median is better
imageDataPath = '/data/gpfs/projects/punim1836/Data/raw/CT_MRI_Pre_Post_add/CT Pre/'; % change as required
augPath = '/data/gpfs/projects/punim1836/Data/raw/aug/'; % change as required
nAug = 1;
refSc = 0.2604;

% data tags
imageTag = 'Pre'; % Pre (for CT) or MR, and Post (for CT)
roiTags = {'LLSCC ant', 'LLSCC post', 'RLSCC ant', 'RLSCC post'};

if strcmp(imageTag, 'Pre') | strcmp(imageTag, 'Post')
    imageTypeTag = 'CT';
    minMax = 3092;
else
    imageTypeTag = 'MR';
    minMax = 786;
end

% get image files
imageFiles = dir(imageDataPath);
imageFiles = {imageFiles.name};
imageFiles = imageFiles(3:end);

% get roi info
[roiNum, roiStr] = xlsread(roiFile);
roiNum = roiNum(:, 1:3); % select the median
roiStr = roiStr(3:end, 1:3);

patList = roiStr(:, 1);
patIdx = find(~cellfun(@isempty, roiStr(:, 1)));
nPat = numel(patIdx);
patList = patList(patIdx);

imgNameList = roiStr(:, 1);

%for pIdx = 15:nPat
for pIdx = 1:nPat
    patName = patList{pIdx};    
    imgIdx = find(strcmp(imgNameList, patName));
    ptNames = roiStr(imgIdx:imgIdx+3, 2);
    ptsOrig = roiNum(imgIdx:imgIdx+3, :);
    pts = zeros(4, 3);

    % re-order if required
    for tIdx = 1:4
        rtIdx = find(strcmp(ptNames, roiTags{tIdx}));
        pts(tIdx, :) = ptsOrig(rtIdx, :);
    end

    % see if corresponding volume exists
    ifIdx = find(strcmp(imageFiles, [patName, ' ', imageTag]));
    if isempty(ifIdx)
        disp('Image does not exist');
        continue;
    end

    try
        % get volume data
        dicomPath = [imageDataPath imageFiles{ifIdx}];
        vol = dicomreadVolume(dicomPath);
    catch
        disp('Cannot load image');
        continue;
    end

    fprintf("loaded dicom volume for patient: %s +++++++++++\n", patName)
    vol(find(vol > minMax)) = minMax;
    vol = squeeze(vol);
    vol = double(vol);
    % vol = rescale(vol);
    sz = size(vol);

    % load dicom file
    dicomFiles = dir(dicomPath);
    dicomFiles = {dicomFiles.name};
    dicomFiles = dicomFiles(3:end);
    meta = dicominfo([dicomPath, '/', dicomFiles{1}]);
    
    % resize volume so that voxels are square
    % sp = [meta.PixelSpacing(1), meta.PixelSpacing(2), meta.SliceThickness];
    % sc = sp / refSc;
    % sp = sp ./ sc;
    % sz = round(sz .* sc);
    % [vol, pts] = rescaleData(vol, pts, sz);

    saveDividedAugmentedPtData(vol, pts, nAug, patName);
    %saveAugmentedPtData(vol, pts, nAug, patName);
    %saveAugmentedData(vol, pts, rotAng, sp, nAug, inPath, outAngPath, outOrigPath, pIdx, nameStr);
    fprintf("finished augmentation for patient: %s ------------------\n", patName)
end

%coordModel;

