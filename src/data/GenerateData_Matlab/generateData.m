clear all;
close all;

% reset random value generator
rng default

roiFile = '/Users/achs/PhD/code/CT-MRI_LandmarkDetection/data/processed/Y/ROI_MR_6.xlsx'; % median is better
%roiFile = '.\Data\ROI\ROI JM.xlsx';
%strFile = 'F:\Code\CT-MRI_LandmarkDetection\Resources\nameStrings.xlsx';
imageDataPath = '/data/gpfs/projects/punim1836/Data/raw/CT_MRI_Pre_Post/'; % change as required
augPath = '/data/gpfs/projects/punim1836/Data/raw/aug/'; % change as required
trainPath = [augPath 're_aug/'];
valPath = [augPath 're_aug/'];
testPath = [augPath 're_aug/'];
inputFolder = 'Input/';
outputFolder = 'Output/';
outputAngFolder = 'OutputAng/';
outputOrigFolder = 'OutputOrig/';
%resPath = '.\Results\';
trainRat = 0.7;
valRat = 0.1;
rotAng = 20;
nAug = 50;
refSc = 0.2604;

% data tags
imageTag = 'Pre'; % Pre (for CT) or MR
roiTags = {'LLSCC ant', 'LLSCC post', 'RLSCC ant', 'RLSCC post'};

if strcmp(imageTag, 'Pre')
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

% get naming strings
%[~, nameStr] = xlsread(strFile);

% get roi info
[roiNum, roiStr] = xlsread(roiFile);
roiNum = roiNum(:, 1:3); % select the median
roiStr = roiStr(3:end, 1:3);

patList = roiStr(:, 1);
patIdx = find(~cellfun(@isempty, roiStr(:, 1)));
nPat = numel(patIdx);
patList = patList(patIdx);

% separate images into sets
nTrain = round(nPat * trainRat);
nVal = round(nPat * valRat);
nTest = nPat - nTrain - nVal;
trainIdx = randperm(nPat, nTrain);
diffIdx = setdiff(1:nPat, trainIdx);
valIdx = diffIdx(randperm(numel(diffIdx), nVal));
testIdx = setdiff(diffIdx, valIdx);

imgNameList = roiStr(:, 1);

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
    vol = rescale(vol);
    sz = size(vol);

    % load dicom file
    dicomFiles = dir(dicomPath);
    dicomFiles = {dicomFiles.name};
    dicomFiles = dicomFiles(3:end);
    meta = dicominfo([dicomPath, '/', dicomFiles{1}]);
    
    % resize volume so that voxels are square
%     sp = [meta.PixelSpacing, meta.SliceThickness];
%     sc = sp / refSc;
%     sp = sp ./ sc;
%     sz = round(sz .* sc);
%     [vol, pts] = rescaleData(vol, pts, sz);

    if ~isempty(find(trainIdx == pIdx))
        inPath = [trainPath, inputFolder];
        %outAngPath = [trainPath, outputAngFolder];
        %outOrigPath = [trainPath, outputOrigFolder]; 
        outPath = [trainPath, outputFolder];
    elseif ~isempty(find(valIdx == pIdx))
        inPath = [valPath, inputFolder];
        %outAngPath = [valPath, outputAngFolder];
        %outOrigPath = [valPath, outputOrigFolder];  
        outPath = [valPath, outputFolder];
    else
        inPath = [testPath, inputFolder];
        %outAngPath = [testPath, outputAngFolder];
        %outOrigPath = [testPath, outputOrigFolder];  
        outPath = [testPath, outputFolder];
    end

    saveDividedAugmentedPtData(vol, pts, nAug, patName);
    %saveAugmentedPtData(vol, pts, nAug, inPath, outPath, patName);
    %saveAugmentedData(vol, pts, rotAng, sp, nAug, inPath, outAngPath, outOrigPath, pIdx, nameStr);
    fprintf("finished augmentation for patient: %s ------------------\n", patName)
end

%coordModel;

