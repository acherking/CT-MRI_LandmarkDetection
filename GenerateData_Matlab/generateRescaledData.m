clear all

imgSize1 = [176 176 48];
imgSize2 = [240 240 64];
imgSize3 = [320 320 96];
imgSize4 = [480 480 128];
imgSize5 = [560 560 160];

desDir1 = '/Volumes/Shawn_SSD/PhD/Project/Date/augmentation_rescaled_data/176_176_48/';
desDir2 = '/Volumes/Shawn_SSD/PhD/Project/Date/augmentation_rescaled_data/240_240_64/';
desDir3 = '/Volumes/Shawn_SSD/PhD/Project/Date/augmentation_rescaled_data/320_320_96/';
desDir4 = '/Volumes/Shawn_SSD/PhD/Project/Date/augmentation_rescaled_data/480_480_128/';
desDir5 = '/Volumes/Shawn_SSD/PhD/Project/Date/augmentation_rescaled_data/560_560_160/';

patList = {'AH', 'AZ', 'DE', 'DM', 'DM2', 'DGL', 'FA', 'GE', 'GM', 'GP', 'HB', 'HH', 'JH', 'JM', 'LG', 'LP', 'MJ', 'NV', 'PH', 'SM' };


orig_data_dir = '/Volumes/Shawn_HDD/PhD/Project/Date/augmentation_from_matlab/original_augmentation_data/';

for ptIdx = 1:length(patList)
    ptName = patList{ptIdx};
    for dataIdx = 6:50
        orig_data_file = [orig_data_dir ptName '_aug_' num2str(dataIdx) '.mat'];
        data = load(orig_data_file, 'augVol', 'augPts');
        resize_volume(ptName, data.augVol, data.augPts, imgSize1, dataIdx, desDir1)
        %resize_volume(ptName, data.augVol, data.augPts, imgSize2, dataIdx, desDir2)
        %resize_volume(ptName, data.augVol, data.augPts, imgSize3, dataIdx, desDir3)
        %resize_volume(ptName, data.augVol, data.augPts, imgSize4, dataIdx, desDir4)
        %resize_volume(ptName, data.augVol, data.augPts, imgSize5, dataIdx, desDir5)
    end
end