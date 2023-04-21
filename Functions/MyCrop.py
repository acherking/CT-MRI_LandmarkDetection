import numpy as np


def find_borders(volume_shape, anchor, crop_size):
    """
    crop_size: ((x_d, x_a), (y_d, y_a), (z_d, z_a))
        x_d is the length from the centre of the given points to the descending direction of axis x, include the given points
        x_a is the length from the centre of the given points to the ascending direction of axis x
        ...
    """
    ((x_d, x_a), (y_d, y_a), (z_d, z_a)) = crop_size

    # find the border on axis x
    x_start = 0
    x_idx_min = anchor[0] - (x_d - 1)
    x_length = x_d + x_a
    if x_idx_min < 0:
        x_length = x_length - abs(x_idx_min)
        x_start = abs(x_idx_min)
        x_idx_min = 0
    x_idx_max = anchor[0] + x_a
    if x_idx_max > (volume_shape[0] - 1):
        x_length = x_length - (x_idx_max - (volume_shape[0] - 1))
        x_idx_max = volume_shape[0] - 1

    # find the border on axis y
    y_start = 0
    y_idx_min = anchor[1] - (y_d - 1)
    y_length = y_d + y_a
    if y_idx_min < 0:
        y_length = y_length - abs(y_idx_min)
        y_start = abs(y_idx_min)
        y_idx_min = 0
    y_idx_max = anchor[1] + y_a
    if y_idx_max > (volume_shape[1] - 1):
        y_length = y_length - (y_idx_max - (volume_shape[1] - 1))
        y_idx_max = volume_shape[1] - 1

    # find the border on axis z
    z_start = 0
    z_idx_min = anchor[2] - (z_d - 1)
    z_length = z_d + z_a
    if z_idx_min < 0:
        z_length = z_length - abs(z_idx_min)
        z_start = abs(z_idx_min)
        z_idx_min = 0
    z_idx_max = anchor[2] + z_a
    if z_idx_max > (volume_shape[2] - 1):
        z_length = z_length - (z_idx_max - (volume_shape[2] - 1))
        z_idx_max = volume_shape[2] - 1

    return x_start, x_length, x_idx_min, y_start, y_length, y_idx_min, z_start, z_length, z_idx_min


def crop(volume, points, crop_size):
    """
    crop_size: ((x_d, x_a), (y_d, y_a), (z_d, z_a))
        x_d is the length from the centre of the given points to the descending direction of axis x, include the given points
        x_a is the length from the centre of the given points to the ascending direction of axis x
        ...
    """
    fill_value = np.min(volume)
    ((x_d, x_a), (y_d, y_a), (z_d, z_a)) = crop_size
    # initialize the cropped_volume with the minimal value in the volume
    cropped_volume = np.ones((x_d + x_a, y_d + y_a, z_d + z_a)) * fill_value

    centre = np.average(points, axis=0).astype(int)

    x_start, x_length, x_idx_min, \
        y_start, y_length, y_idx_min, \
        z_start, z_length, z_idx_min = find_borders(volume.shape, centre, crop_size)

    # crop the volume
    cropped_volume[x_start:(x_start + x_length), y_start:(y_start + y_length), z_start:(z_start + z_length)] = \
        volume[x_idx_min:(x_idx_min + x_length), y_idx_min:(y_idx_min + y_length), z_idx_min:(z_idx_min + z_length)]
    # cropped length, used to relocate the cropped points & find the point from the original whole volume
    cropped_length = np.ones(points.shape) * np.array([x_idx_min, y_idx_min, z_idx_min]) - \
                     np.ones(points.shape) * np.array([x_start, y_start, z_start])
    # relocate the points
    cropped_points = points - cropped_length

    return cropped_volume, cropped_points, cropped_length


def crop_volume(volume, points, crop_size=((50, 50), (50, 50), (50, 50))):
    """
    Crop the landmark areas for left and right ears separately
    The outputs are two cubic volume 100*100*100 (may change in the future).
    Input:  1. Volume: the original volume
            2. Points: LLSCC ant/post, RLSCC ant/post

    Output: 1. left_area_volume, left_landmark_points, left_cropped_length
            2. right_area_volume, right_landmark_points, right_cropped_length
    """
    # crop_size = ((50, 50), (50, 50), (50, 50))
    # points coordinate is (x, y, z), swap to (y, x, z) to cooperate the volume (row, clown, slice)
    points = points[:, [1, 0, 2]]
    left_area, left_landmarks, left_cropped_length = crop(volume, points[0:2], crop_size)
    left_landmarks = left_landmarks[:, [1, 0, 2]]
    left_cropped_length = left_cropped_length[:, [1, 0, 2]]
    right_area, right_landmarks, right_cropped_length = crop(volume, points[2:4], crop_size)
    right_landmarks = right_landmarks[:, [1, 0, 2]]
    right_cropped_length = right_cropped_length[:, [1, 0, 2]]

    return left_area, left_landmarks, left_cropped_length, right_area, right_landmarks, right_cropped_length


def flip_volume(volume, points):
    volume_s = volume.shape
    flip_v = np.fliplr(volume)
    flip_p = np.copy(points)
    flip_p[:, [0]] = np.ones(flip_p[:, [0]].shape) * (volume_s[1] - 1) - flip_p[:, [0]]

    return flip_v, flip_p


def distance_from_border(volume_shape, points, anchor, crop_size=((50, 50), (50, 50), (50, 50))):
    """
    crop_size: ((x_d, x_a), (y_d, y_a), (z_d, z_a))
        x_d is the length from the centre of the given points to the descending direction of axis x, include the given points
        x_a is the length from the centre of the given points to the ascending direction of axis x
        ...
    points: two 3D points (2*3)
    :return
    (2*2*3) --> (2 points, descending&ascending, x&y&z)
    """

    x_start, x_length, x_idx_min, \
        y_start, y_length, y_idx_min, \
        z_start, z_length, z_idx_min = find_borders(volume_shape, anchor, crop_size)

    cropped_length = np.ones(points.shape) * np.array([x_idx_min, y_idx_min, z_idx_min]) - \
                     np.ones(points.shape) * np.array([x_start, y_start, z_start])

    # points' distances to the borders, to the descending direction (just like the coordinate in the cropped volume:))
    points_d = points - cropped_length
    points_a = np.ones(points.shape) * np.array([x_length, y_length, z_length]) - points_d

    # concatenate the results
    points_d = np.expand_dims(points_d, axis=1)
    points_a = np.expand_dims(points_a, axis=1)
    ret = np.concatenate((points_d, points_a), axis=1)

    return ret
