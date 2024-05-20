%% Read Original Augmentation Volume
% patients = ["AH" "AZ" 'DE' 'DM' 'DM2' 'DGL' 'FA' 'GE' 'GM' 'GP' 'HB' 'HH' ...
%     'JH' 'JM' 'LG' 'LP' 'MJ' 'NV' 'PH' 'SM'];
% 
% size_all = [];
% for pat_id = 1:20
%     pat = patList(pat_id);
%     for idx = 1:50
%         idx_str = string(idx);
%         file_path = "F:\Data\augmentation_exp\original_divided_augmentation\" + pat + "_aug_" + idx_str + ".mat";
%         disp(file_path);
%         size = load(file_path, "augVolSize").augVolSize();
%         size_all = cat(1, size_all, size);
%     end
% end
% 
% size_max = max(size_all);

%% load all
patients = ["AH" "AZ" 'DE' 'DM' 'DM2' 'DGL' 'FA' 'GE' 'GM' 'GP' 'HB' 'HH' ...
    'JH' 'JM' 'LG' 'LP' 'MJ' 'NV' 'PH' 'SM'];

size_max = [1262, 1252, 684];

size_all = [];
for pat_id = 1:20
    pat = patList(pat_id);
    for idx = 1:50
        idx_str = string(idx);
        file_path = "F:\Data\augmentation_exp\original_divided_augmentation\" + pat + "_aug_" + idx_str + ".mat";
        disp(file_path);
        load_data = load(file_path, "augPts", "augVolSize");
        vol = load_data.augVol;
        pts = load_data.augPts;
        vol_size = load_data.augVolSize;

        padding = size_max-vol_size;
        surround = floor(padding/2);
        post = mod(padding, 2);

        vol_pad = padarray(vol, surround);
        vol_pad = padarray(vol_pad, post, 0, 'post');
        pts_pad = pts + [surround(2), surround(1), surround(3)];

        % Resize and Save
        halfXsize = round(size_max(2) / 2);

        padLeftPts = pts_pad(1:2, :);
        padLeftPts(:, 1) = padLeftPts(:, 1) - halfXsize;
        padRightPts = pts_pad(3:4, :);
        padRightPts(:, 1) = halfXsize - padRightPts(:, 1) + 1;

        padLeftVol = vol_pad(:, (halfXsize+1):end, :);
        padRightVol = vol_pad(:, 1:halfXsize, :);
        padRightVol = flip(padRightVol, 2);

        padLeftVolSize = size(padLeftVol);
        padRightVolSize = size(padRightVol);


        patName = pat{1};
        strIdx = idx_str;
        origBase = "F:\Data\augmentation_exp";

%         augFile = origBase + "\original_divided_augmentation_normalize\" + patName + '_aug_' + strIdx + '.mat';
%         save(augFile, 'vol_pad', 'pts_pad', 'size_max', '-v7.3');
% 
%         fprintf("Saved augmentation vol for patient: %s -- %d \n To Path: %s\n", patName, idx, augFile)


        imgSize = [176 88 48; 176 88 96].';
        % reduce size
        for s = imgSize
            sizeT = s.';
            [padLeftVolRescaled, padLeftPtsRescaled] = rescaleData(padLeftVol, padLeftPts, sizeT);
            [padRightVolRescaled, padRightPtsRescaled] = rescaleData(padRightVol, padRightPts, sizeT);

            padLeftVolRescaledSize = size(padLeftVolRescaled);
            padRightVolRescaledSize = size(padRightVolRescaled);

            % in mm
            oriRes = [0.15, 0.15, 0.15];

            leftScale = padLeftVolRescaledSize ./ padLeftVolSize;
            leftRes = oriRes ./ leftScale;

            rightScale = padRightVolRescaledSize ./ padRightVolSize;
            rightRes = oriRes ./ rightScale;

            strSize = num2str(sizeT(1)) + "x" + num2str(sizeT(2)) + "x" + num2str(sizeT(3));
            inPath = origBase + "\reduce_size_normalize\" + strSize + "\";
            padRescaledFile = inPath + patName + '_' + strSize + '_' + strIdx + '.mat';

            % save to datastores
            save(padRescaledFile, 'padLeftVolRescaled', 'padLeftPtsRescaled', 'leftRes', 'padLeftVolSize', ...
                'padRightVolRescaled', 'padRightPtsRescaled', 'rightRes', 'padRightVolSize','-v7.3');

            fprintf("Saved augmentation vol for patient: %s -- %d \n To Path: %s\n", patName, idx, padRescaledFile);
        end
        
    end
end