base_dir = "/data/gpfs/projects/punim1836/CT-MRI_Registration/src/matlab/for_Shawn/Data/CT/org/";
save_dir = "/data/gpfs/projects/punim1836/CT-MRI_Registration/src/matlab/for_Shawn/Data/CT/org/res_02604/";
names = ["GM" "GE" "AZ" "LP"];
sp = [0.15, 0.15, 0.15];
sc = sp / 0.2604;
for name = names
    file_name = name + "_aug_1.mat";
    load(base_dir + file_name);
    sz = round(augVolSize .* sc);
    [vol, pts] = rescaleData(augVol, augPts, sz);
    save(save_dir+file_name, 'vol', 'pts', '-v7.3');
end