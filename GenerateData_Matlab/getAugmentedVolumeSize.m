clear all

imgSize1 = [176 176 48];
imgSize2 = [240 240 64];
imgSize3 = [320 320 96];
imgSize4 = [480 480 128];
imgSize5 = [560 560 160];

desDir1 = 'F:\Data\rescaled_data\176_176_48_PD\';
desDir2 = 'F:\Data\rescaled_data\240_240_64_PD\';
desDir3 = 'F:\Data\rescaled_data\320_320_96_PD\';
desDir4 = 'F:\Data\rescaled_data\480_480_128_PD\';

patList = {'AH', 'AZ', 'DE', 'DM', 'DM2', 'DGL', 'FA', 'GE', 'GM', 'GP', 'HB', 'HH', 'JH', 'JM', 'LG', 'LP', 'MJ', 'NV', 'PH', 'SM' };

orig_data_dir = 'F:\Data\original_augmentation_data\';
des_data_dir = 'F:\Data\rescaled_data\original_augmentation_volume_size\';

for ptIdx = 1:length(patList)
    ptName = patList{ptIdx};
    for dataIdx = 1:50
        orig_data_file = [orig_data_dir ptName '_aug_' num2str(dataIdx) '.mat'];
        des_data_file = [des_data_dir ptName '_augSize_' num2str(dataIdx) '.mat'];
        data = load(orig_data_file, 'augVol');
        volumeSize = size(data.augVol);
        %save(des_data_file, 'volumeSize', '-v7.3');
        %appendPixelDistance(ptName, volumeSize, imgSize1, dataIdx, desDir1)
        %appendPixelDistance(ptName, volumeSize, imgSize2, dataIdx, desDir2)
        %appendPixelDistance(ptName, volumeSize, imgSize3, dataIdx, desDir3)
        appendPixelDistance(ptName, volumeSize, imgSize4, dataIdx, desDir4)
    end
end
