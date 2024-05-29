import numpy as np


def find_borders(volume_shape, anchor, crop_size):
    """
    crop_size: ((x_d, x_a), (y_d, y_a), (z_d, z_a))
        x_d is the length from the centre of the given points to the descending direction of axis x, include the given points
        x_a is the length from the centre of the given points to the ascending direction of axis x
        ...
    anchor: (num of dimensions)
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


def crop_shape(volume_shape, points, crop_size):
    """
    crop_size: ((x_d, x_a), (y_d, y_a), (z_d, z_a))
        x_d is the length from the centre of the given points to the descending direction of axis x, include the given points
        x_a is the length from the centre of the given points to the ascending direction of axis x
        ...
    """
    ((x_d, x_a), (y_d, y_a), (z_d, z_a)) = crop_size

    centre = np.average(points, axis=0).astype(int)

    x_start, x_length, x_idx_min, \
        y_start, y_length, y_idx_min, \
        z_start, z_length, z_idx_min = find_borders(volume_shape, centre, crop_size)

    # cropped length, used to relocate the cropped points & find the point from the original whole volume
    cropped_length = np.ones(points.shape) * np.array([x_idx_min, y_idx_min, z_idx_min]) - \
                     np.ones(points.shape) * np.array([x_start, y_start, z_start])
    # relocate the points
    cropped_points = points - cropped_length

    return cropped_points, cropped_length


def crop_volume_shape(volume_shape, points, crop_size=((50, 50), (50, 50), (50, 50))):
    """
    Crop the landmark areas for left and right ears separately
    The outputs are two cubic volume 100*100*100 (may change in the future).
    Input:  1. Volume_shape: the original volume's shape
            2. Points: LLSCC ant/post, RLSCC ant/post

    Output: 1. left_landmark_points, left_cropped_length
            2. right_landmark_points, right_cropped_length
    """
    # crop_size = ((50, 50), (50, 50), (50, 50))
    # points coordinate is (x, y, z), swap to (y, x, z) to cooperate the volume (row, clown, slice)
    points = points[:, [1, 0, 2]]
    left_landmarks, left_cropped_length = crop_shape(volume_shape, points[0:2], crop_size)
    left_landmarks = left_landmarks[:, [1, 0, 2]]
    left_cropped_length = left_cropped_length[:, [1, 0, 2]]
    right_landmarks, right_cropped_length = crop_shape(volume_shape, points[2:4], crop_size)
    right_landmarks = right_landmarks[:, [1, 0, 2]]
    right_cropped_length = right_cropped_length[:, [1, 0, 2]]

    return left_landmarks, left_cropped_length, right_landmarks, right_cropped_length


def crop(volume, points, anchor, crop_size):
    """
    crop_size: ((row_d, row_a), (column_d, column_a), (slice_d, slice_a))
        row_d is the length from the centre of the given points to the descending direction of axis x, include the given points
        row_a is the length from the centre of the given points to the ascending direction of axis x
        ...
    """
    fill_value = np.min(volume)
    ((row_d, row_a), (column_d, column_a), (slice_d, slice_a)) = crop_size
    # initialize the cropped_volume with the minimal value in the volume
    cropped_volume = np.ones((row_d + row_a, column_d + column_a, slice_d + slice_a)) * fill_value

    if anchor is not None:
        centre = anchor.astype(int)
    else:
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


def crop_volume_anchor(volume, points, anchor=None, crop_size=((50, 50), (50, 50), (50, 50))):
    """
    Crop the landmark areas for left and right ears separately
    The outputs are two cubic volume 100*100*100 (may change in the future).
    Input:  1. Volume: the original volume
            2. Points: LLSCC ant/post, RLSCC ant/post
            3. anchors: Left centre & Right centre, (2*3)

    Output: 1. left_area_volume, left_landmark_points, left_cropped_length
            2. right_area_volume, right_landmark_points, right_cropped_length
    """
    if anchor is not None:
        left_anchor = anchor[0, [1, 0, 2]]
        right_anchor = anchor[1, [1, 0, 2]]
    else:
        left_anchor = None
        right_anchor = None
    # crop_size = ((50, 50), (50, 50), (50, 50))
    # points coordinate is (x, y, z), swap to (y, x, z) to cooperate the volume (row, clown, slice)
    points = points[:, [1, 0, 2]]
    left_area, left_landmarks, left_cropped_length = crop(volume, points[0:2], left_anchor, crop_size)
    left_landmarks = left_landmarks[:, [1, 0, 2]]
    left_cropped_length = left_cropped_length[:, [1, 0, 2]]
    right_area, right_landmarks, right_cropped_length = crop(volume, points[2:4], right_anchor, crop_size)
    right_landmarks = right_landmarks[:, [1, 0, 2]]
    right_cropped_length = right_cropped_length[:, [1, 0, 2]]

    return left_area, left_landmarks, left_cropped_length, right_area, right_landmarks, right_cropped_length


# x_volumes_org: (instance_num, row_num, column_num, slice_num, 1)
# y_landmarks_org: (instance_num, landmarks_num ,dimensions_num)
# length_org: (instance_num, landmarks_num, dimensions_num)
# crop_layers: ndarray shape(3*2), [[row_ascending, row_descending], [column_a, column_d], [slice_a, slice_d]]
def crop_outside_layers(x_volumes_org, y_landmarks_org, length_org, crop_layers, keep_blank=True):
    x_dataset = np.copy(x_volumes_org)
    y_dataset = np.copy(y_landmarks_org)
    length_dataset = np.copy(length_org)

    instances_num = x_dataset.shape[0]

    row_num = x_dataset.shape[1]
    column_num = x_dataset.shape[2]
    slice_num = x_dataset.shape[3]

    if keep_blank:
        fill_val = np.min(x_dataset)
        x_dataset_corroded = np.ones(x_dataset.shape) * fill_val
        x_dataset_corroded[:,
        crop_layers[0][0]:(row_num - crop_layers[0][1]),
        crop_layers[1][0]:(column_num - crop_layers[1][1]),
        crop_layers[2][0]:(slice_num - crop_layers[2][1]), :] = \
            x_dataset[:,
            crop_layers[0][0]:(row_num - crop_layers[0][1]),
            crop_layers[1][0]:(column_num - crop_layers[1][1]),
            crop_layers[2][0]:(slice_num - crop_layers[2][1]), :]
        # to make the return consistent
        x_dataset = x_dataset_corroded
    else:
        x_dataset = x_dataset[:,
                    crop_layers[0][0]:(row_num - crop_layers[0][1]),
                    crop_layers[1][0]:(column_num - crop_layers[1][1]),
                    crop_layers[2][0]:(slice_num - crop_layers[2][1]), :]
        y_dataset = y_dataset - [crop_layers[1, 0], crop_layers[0, 0], crop_layers[2, 0]]
        # y_dataset = y_dataset.astype('float32')
        # left ear
        length_dataset[range(0, instances_num, 2)] = \
            length_dataset[range(0, instances_num, 2)] + [crop_layers[1, 0], crop_layers[0, 0], crop_layers[2, 0]]
        # right ear, because of the flip
        length_dataset[range(1, instances_num, 2)] = \
            length_dataset[range(1, instances_num, 2)] + [crop_layers[1, 1], crop_layers[0, 0], crop_layers[2, 0]]
        # length_dataset = length_dataset.astype('float32')

    return x_dataset, y_dataset, length_dataset


def crop_outside_layers_no_length(x_volumes_org, y_landmarks_org, crop_layers, keep_blank=True):
    x_dataset = np.copy(x_volumes_org)
    y_dataset = np.copy(y_landmarks_org)

    instances_num = x_dataset.shape[0]

    row_num = x_dataset.shape[1]
    column_num = x_dataset.shape[2]
    slice_num = x_dataset.shape[3]

    if keep_blank:
        fill_val = np.min(x_dataset)
        x_dataset_corroded = np.ones(x_dataset.shape) * fill_val
        x_dataset_corroded[:,
            crop_layers[0][0]:(row_num - crop_layers[0][1]),
            crop_layers[1][0]:(column_num - crop_layers[1][1]),
            crop_layers[2][0]:(slice_num - crop_layers[2][1]), :] = \
            x_dataset[:,
            crop_layers[0][0]:(row_num - crop_layers[0][1]),
            crop_layers[1][0]:(column_num - crop_layers[1][1]),
            crop_layers[2][0]:(slice_num - crop_layers[2][1]), :]
        # to make the return consistent
        x_dataset = x_dataset_corroded
    else:
        x_dataset = x_dataset[:,
                    crop_layers[0][0]:(row_num - crop_layers[0][1]),
                    crop_layers[1][0]:(column_num - crop_layers[1][1]),
                    crop_layers[2][0]:(slice_num - crop_layers[2][1]), :]
        y_dataset = y_dataset - [crop_layers[1, 0], crop_layers[0, 0], crop_layers[2, 0]]

    return x_dataset, y_dataset


# x_volumes_org: (instance_num, row_num, column_num, slice_num, 1)
# y_landmarks_org: (instance_num, landmarks_num ,dimensions_num)
# length_org: (instance_num, landmarks_num, dimensions_num)
# centre_shift: (instance_num, 1, dimensions_num); in mm; '+' --> shift in descending order, '-' opposite
def crop_outside_layers_trans(x_volumes_org, y_landmarks_org, length_org,
                              centre_shift, target_shape=(150, 150, 100)):
    x_dataset = x_volumes_org
    y_dataset = y_landmarks_org
    length_dataset = length_org

    # for convenience
    # centre_shift = centre_shift.reshape(centre_shift.shape[0], 1, centre_shift.shape[1])
    centre_shift = np.repeat(centre_shift, 2, axis=1)

    # swap x and y
    y_dataset = y_dataset[:, :, [1, 0, 2]]
    length_dataset = length_dataset[:, :, [1, 0, 2]]
    centre_shift = centre_shift[:, :, [1, 0, 2]]

    # debug
    print("***********Original Y **********")
    print(y_dataset[0:10])
    print("***********Original length **********")
    print(length_dataset[0:10])
    print("***********Centre Shift **********")
    print(centre_shift[0:10])

    # calculate the crop parameters
    row_start_base = np.ceil((x_dataset.shape[1] - target_shape[0]) / 2)
    column_start_base = np.ceil((x_dataset.shape[2] - target_shape[1]) / 2)
    slice_start_base = np.ceil((x_dataset.shape[3] - target_shape[2]) / 2)
    crop_start_array = np.ones(y_dataset.shape) * [row_start_base, column_start_base, slice_start_base]
    crop_start_array = (crop_start_array - centre_shift / 0.15).astype(int)
    crop_start_array = np.where(crop_start_array < 0, 0, crop_start_array)
    crop_start_array = np.where(crop_start_array > 50, 50, crop_start_array)

    # calculate new y_dataset and length_dataset (after crop)
    y_dataset = y_dataset - crop_start_array
    ## left ear
    length_dataset[range(0, 2000, 2)] = length_dataset[range(0, 2000, 2)] + crop_start_array[range(0, 2000, 2)]
    ## right ear, because of the flip
    right_ear_length_shift = np.copy(crop_start_array[range(1, 2000, 2)])
    right_ear_length_shift[:, :, 1] = np.ones(right_ear_length_shift[:, :, 1].shape) * \
                                      (x_dataset.shape[2] - target_shape[1]) - right_ear_length_shift[:, :, 1]
    length_dataset[range(1, 2000, 2)] = length_dataset[range(1, 2000, 2)] + right_ear_length_shift

    # debug
    print("***********New Y **********")
    print(y_dataset[0:10])
    print("***********New Length **********")
    print(length_dataset[0:10])

    # swap back x and y; for Y_dataset, length_dataset
    y_dataset = y_dataset[:, :, [1, 0, 2]]
    length_dataset = length_dataset[:, :, [1, 0, 2]]

    # calculate the cropped volume
    fill_val = np.min(x_dataset)
    x_dataset_corroded = np.ones((x_dataset.shape[0], 150, 150, 100, 1)) * fill_val

    for idx in range(0, x_dataset.shape[0]):
        if idx % 100 == 0:
            print("Crop volume: ", idx)
        x_dataset_corroded[idx] = x_dataset[idx,
                                  crop_start_array[idx, 0, 0]:(crop_start_array[idx, 0, 0] + target_shape[0]),
                                  crop_start_array[idx, 0, 1]:(crop_start_array[idx, 0, 1] + target_shape[1]),
                                  crop_start_array[idx, 0, 2]:(crop_start_array[idx, 0, 2] + target_shape[2]), :]

    return x_dataset_corroded, y_dataset, length_dataset


# points: (num of points, num of dimensions)
def flip_volume(volume, points):
    volume_s = volume.shape
    flip_v = np.fliplr(volume)
    flip_p = np.copy(points)
    flip_p[:, [0]] = np.ones(flip_p[:, [0]].shape) * (volume_s[1] + 1) - flip_p[:, [0]]

    return flip_v, flip_p


def flip_volume_shape(volume_shape, points):
    flip_p = np.copy(points)
    flip_p[:, [0]] = np.ones(flip_p[:, [0]].shape) * (volume_shape[1] + 1) - flip_p[:, [0]]

    return flip_p


def distance_from_border(volume_shape, points, anchor, crop_size=((50, 50), (50, 50), (50, 50))):
    """
    crop_size: ((x_d, x_a), (y_d, y_a), (z_d, z_a))
        x_d is the length from the centre of the given points to the descending direction of axis x, include the given points
        x_a is the length from the centre of the given points to the ascending direction of axis x
        ...
    points: two 3D points (2*3)
    anchor: (num of dimensions,)
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


def cut_flip_volume(volume, points):
    """
    lower_side: looks is left side, but actually is RLSCC ant/post
    higher_side: looks is right side, but actually is LLSCC ant/post
    Points: LLSCC ant/post, RLSCC ant/post
    """
    column_centre = volume.shape[1] / 2
    lower_side_end = np.ceil(column_centre).astype(int)
    higher_side_start = np.floor(column_centre).astype(int)

    lower_side_volume = volume[:, 0:lower_side_end, :]
    higher_side_volume = volume[:, higher_side_start:, :]

    left_landmarks = np.copy(points[0:2])
    left_landmarks_cut = lower_side_end - volume.shape[1] % 2
    left_landmarks[:, 0] = left_landmarks[:, 0] - left_landmarks_cut
    right_landmarks = np.copy(points[2:4])

    # check if left/right landmarks are involved in each side volume
    if left_landmarks[:, 0].any() < 0:
        print("Left landmarks outside the divided volume.")
        return
    if right_landmarks[:, 0].any() >= lower_side_end:
        print("Right landmarks outside the divided volume.")
        return

    return higher_side_volume, left_landmarks, lower_side_volume, right_landmarks, left_landmarks_cut


def cut_flip_volume_shape(volume_shape):
    column_centre = volume_shape[1] / 2
    lower_side_end = np.ceil(column_centre).astype(int)

    new_shape = (volume_shape[0], lower_side_end, volume_shape[1])
    left_landmarks_cut = lower_side_end - volume_shape[1] % 2

    return new_shape, left_landmarks_cut
