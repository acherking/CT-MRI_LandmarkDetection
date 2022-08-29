file_vol= 'test_save_vol.mat'
file_pts = 'test_save_pts.mat'

v = augVol
save(file_vol, 'v', '-v7.3');

v = augPts
save(file_pts, 'v')

