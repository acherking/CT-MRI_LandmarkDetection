nAug = 50;

padSize = [943 938 402];
imgSize = [176 176 48];

oriRes = [0.2604, 0.2604, 0.2604];

base_save = "/data/gpfs/projects/punim1836/Data/raw/aug";

base = "/data/gpfs/projects/punim1836/Data/raw/aug/original_augmentation_";
base_modes = ["CT_Pre" 'CT_Post' 'MRI'];

pat_list = ["AH", 'AZ', 'DE', 'DM', 'DM2', 'DGL', 'FA', 'GE', 'GM', 'GP', 'HB', 'HH', 'JH', 'JM', 'LG', 'LP', 'MJ', 'NV', 'PH', 'SM'];

% load
for base_mode = base_modes
	base_dir = base + base_mode;
	fprintf("start: " + base_dir + '\n');
	for pat = pat_list
		fprintf("patient: " + pat + '\n');
		idx = 1;
		while idx <= nAug
			strIdx = string(idx);
			% load data
			filePath = base_dir + '/' + pat + '_aug_' + strIdx + '.mat';
			if exist(filePath, 'file') ~= 2
				break
			end
			
			load(filePath, 'augVol', 'augPts', 'augMask', 'augVolSize')
			% pad
			trans = ceil((padSize - augVolSize)/2);
			startPos = trans + 1;
			endPos = startPos + augVolSize - 1;

			padPts = augPts + [trans(2) trans(1) trans(3)];
			padVol = min(augVol(:)) * ones(padSize);
			padVol(startPos(1):endPos(1), startPos(2):endPos(2), startPos(3):endPos(3)) = augVol;
			padMask = zeros(padSize);
			padMask(startPos(1):endPos(1), startPos(2):endPos(2), startPos(3):endPos(3)) = augMask;
			% rescalse
			[augVolRescaled, augPtsRescaled, augMaskRescaled] = rescaleData(padVol, padPts, imgSize, padMask);

			augVolRescaledSize = size(augVolRescaled);

			scale = augVolRescaledSize ./ padSize;
			res = oriRes ./ scale;

			inPath = base_save + "/reduce_size_" + base_mode + "/176x176x48_res/";
			augRescaledFile = inPath + pat + '_176x176x48_' + strIdx + '.mat';
			save(augRescaledFile, 'augVolRescaled', 'augPtsRescaled', 'res', ...
				            'augVolSize', 'augMaskRescaled', '-v7.3');
			fprintf("Saved augmentation vol for patient: %s -- %d \n To Path: %s\n", pat, idx, augRescaledFile);

			idx = idx + 1;
		end
	end
end
