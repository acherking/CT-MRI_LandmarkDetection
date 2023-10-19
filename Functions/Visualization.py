import numpy

from matplotlib import pyplot
from matplotlib.patches import Circle


# print the Scans with pts: LLSCC ant, LLSCC post, RLSCC ant, RLSCC post
def show_pts(volume, pts, pixel_space):
    y_row = numpy.arange(0.0, volume.shape[0] * pixel_space[0], pixel_space[0])
    x_column = numpy.arange(0.0, volume.shape[1] * pixel_space[1], pixel_space[1])
    z_slice = numpy.arange(0.0, volume.shape[2] * pixel_space[2], pixel_space[2])

    pts = pts - 1

    landmark_radius = volume.shape[1] * pixel_space[1] * 0.016

    fig, axs = pyplot.subplots(2, 2, sharex=True)
    fig.set_dpi(300)

    axs[0][0].set_title("LLSCC ant")
    axs[0][0].set_aspect('equal', 'datalim')
    axs[0][0].pcolormesh(x_column[:], y_row[:], volume[:, :, round(pts[0, 2])], cmap=pyplot.gray())
    llscc_ant = Circle((pts[0, 0] * pixel_space[1], pts[0, 1] * pixel_space[0]), landmark_radius, facecolor='None',
                       edgecolor='r', lw=1)
    axs[0][0].add_patch(llscc_ant)
    axs[0][0].invert_yaxis()

    axs[0][1].set_title("LLSCC post")
    axs[0][1].set_aspect('equal', 'datalim')
    axs[0][1].pcolormesh(x_column[:], y_row[:], volume[:, :, round(pts[1, 2])], cmap=pyplot.gray())
    llscc_post = Circle((pts[1, 0] * pixel_space[1], pts[1, 1] * pixel_space[0]), landmark_radius, facecolor='None',
                        edgecolor='r', lw=1)
    axs[0][1].add_patch(llscc_post)
    axs[0][1].invert_yaxis()

    axs[1][0].set_title("RLSCC ant")
    axs[1][0].set_aspect('equal', 'datalim')
    axs[1][0].pcolormesh(x_column[:], y_row[:], volume[:, :, round(pts[2, 2])], cmap=pyplot.gray())
    rlscc_ant = Circle((pts[2, 0] * pixel_space[1], pts[2, 1] * pixel_space[0]), landmark_radius, facecolor='None',
                       edgecolor='r', lw=1)
    axs[1][0].add_patch(rlscc_ant)
    axs[1][0].invert_yaxis()

    axs[1][1].set_title("RLSCC post")
    axs[1][1].set_aspect('equal', 'datalim')
    axs[1][1].pcolormesh(x_column[:], y_row[:], volume[:, :, round(pts[3, 2])], cmap=pyplot.gray())
    rlscc_post = Circle((pts[3, 0] * pixel_space[1], pts[3, 1] * pixel_space[0]), landmark_radius, facecolor='None',
                        edgecolor='r', lw=1)
    axs[1][1].add_patch(rlscc_post)
    axs[1][1].invert_yaxis()

    pyplot.setp(axs[-1, :], xlabel='(mm)')
    pyplot.setp(axs[:, 0], ylabel='(mm)')

    pyplot.show()


def show_two_landmarks_half_volume(left_volume, left_points, pixel_space):
    #
    y_row = numpy.arange(0.0, left_volume.shape[0] * pixel_space[0], pixel_space[0])
    x_column = numpy.arange(0.0, left_volume.shape[1] * pixel_space[1], pixel_space[1])
    z_slice = numpy.arange(0.0, left_volume.shape[2] * pixel_space[2], pixel_space[2])

    left_points = left_points - 1
    # right_points[:, [0]] = right_points[:, [0]] + 2

    landmark_radius = left_volume.shape[1] * pixel_space[1] * 0.016

    fig, axs = pyplot.subplots(1, 2, sharex=True, constrained_layout=True)
    fig.set_dpi(400)

    axs[0].set_title("LLSCC ant")
    axs[0].set_aspect('equal', 'box')
    im00 = axs[0].pcolormesh(x_column[:], y_row[:], left_volume[:, :, round(left_points[0, 2])], cmap=pyplot.gray())
    llscc_ant = Circle((left_points[0, 0] * pixel_space[1], left_points[0, 1] * pixel_space[0]), landmark_radius, facecolor='None',
                       edgecolor='r', lw=1)
    axs[0].add_patch(llscc_ant)
    axs[0].invert_yaxis()

    fig.colorbar(im00, ax=axs[0])

    axs[1].set_title("LLSCC post")
    axs[1].set_aspect('equal', 'box')
    im01 = axs[1].pcolormesh(x_column[:], y_row[:], left_volume[:, :, round(left_points[1, 2])], cmap=pyplot.gray())
    llscc_post = Circle((left_points[1, 0] * pixel_space[1], left_points[1, 1] * pixel_space[0]), landmark_radius, facecolor='None',
                        edgecolor='r', lw=1)
    axs[1].add_patch(llscc_post)
    axs[1].invert_yaxis()

    fig.colorbar(im01, ax=axs[1])

    pyplot.setp(axs[-1], xlabel='(mm)')
    pyplot.setp(axs[0], ylabel='(mm)')

    pyplot.show()


def show_one_landmark_half_volume(left_volume, left_point, pixel_space):
    #
    y_row = numpy.arange(0.0, left_volume.shape[0] * pixel_space[0], pixel_space[0])
    x_column = numpy.arange(0.0, left_volume.shape[1] * pixel_space[1], pixel_space[1])
    z_slice = numpy.arange(0.0, left_volume.shape[2] * pixel_space[2], pixel_space[2])

    left_point = left_point - 1
    # right_points[:, [0]] = right_points[:, [0]] + 2

    landmark_radius = left_volume.shape[1] * pixel_space[1] * 0.016

    fig, axs = pyplot.subplots(1, 1, sharex=True, constrained_layout=True)
    fig.set_dpi(400)

    axs.set_title("LLSCC ant")
    axs.set_aspect('equal', 'box')
    im00 = axs.pcolormesh(x_column[:], y_row[:], left_volume[:, :, round(left_point[0, 2])], cmap=pyplot.gray())
    llscc_ant = Circle((left_point[0, 0] * pixel_space[1], left_point[0, 1] * pixel_space[0]), landmark_radius, facecolor='None',
                       edgecolor='r', lw=1)
    axs.add_patch(llscc_ant)
    axs.invert_yaxis()

    fig.colorbar(im00, ax=axs)

    pyplot.setp(axs, xlabel='(mm)')
    pyplot.setp(axs, ylabel='(mm)')

    pyplot.show()


def show_two_landmarks(left_volume, left_points, right_volume, right_points, pixel_space):
    #
    y_row = numpy.arange(0.0, left_volume.shape[0] * pixel_space[0], pixel_space[0])
    x_column = numpy.arange(0.0, left_volume.shape[1] * pixel_space[1], pixel_space[1])
    z_slice = numpy.arange(0.0, left_volume.shape[2] * pixel_space[2], pixel_space[2])

    left_points = left_points - 1
    right_points = right_points - 1
    # right_points[:, [0]] = right_points[:, [0]] + 2

    landmark_radius = left_volume.shape[1] * pixel_space[1] * 0.016

    fig, axs = pyplot.subplots(2, 2, sharex=True, constrained_layout=True)
    fig.set_dpi(400)

    axs[0][0].set_title("LLSCC ant")
    axs[0][0].set_aspect('equal', 'box')
    im00 = axs[0][0].pcolormesh(x_column[:], y_row[:], left_volume[:, :, round(left_points[0, 2])], cmap=pyplot.gray())
    llscc_ant = Circle((left_points[0, 0] * pixel_space[1], left_points[0, 1] * pixel_space[0]), landmark_radius, facecolor='None',
                       edgecolor='r', lw=1)
    axs[0][0].add_patch(llscc_ant)
    axs[0][0].invert_yaxis()

    fig.colorbar(im00, ax=axs[0][0])

    axs[0][1].set_title("LLSCC post")
    axs[0][1].set_aspect('equal', 'box')
    im01 = axs[0][1].pcolormesh(x_column[:], y_row[:], left_volume[:, :, round(left_points[1, 2])], cmap=pyplot.gray())
    llscc_post = Circle((left_points[1, 0] * pixel_space[1], left_points[1, 1] * pixel_space[0]), landmark_radius, facecolor='None',
                        edgecolor='r', lw=1)
    axs[0][1].add_patch(llscc_post)
    axs[0][1].invert_yaxis()

    fig.colorbar(im01, ax=axs[0][1])

    axs[1][0].set_title("RLSCC ant")
    axs[1][0].set_aspect('equal', 'box')
    im10 = axs[1][0].pcolormesh(x_column[:], y_row[:], right_volume[:, :, round(right_points[0, 2])], cmap=pyplot.gray())
    rlscc_ant = Circle((right_points[0, 0] * pixel_space[1], right_points[0, 1] * pixel_space[0]), landmark_radius, facecolor='None',
                       edgecolor='r', lw=1)
    axs[1][0].add_patch(rlscc_ant)
    axs[1][0].invert_yaxis()

    fig.colorbar(im10, ax=axs[1][0])

    axs[1][1].set_title("RLSCC post")
    axs[1][1].set_aspect('equal', 'box')
    im11 = axs[1][1].pcolormesh(x_column[:], y_row[:], right_volume[:, :, round(right_points[1, 2])], cmap=pyplot.gray())
    rlscc_post = Circle((right_points[1, 0] * pixel_space[1], right_points[1, 1] * pixel_space[0]), landmark_radius, facecolor='None',
                        edgecolor='r', lw=1)
    axs[1][1].add_patch(rlscc_post)
    axs[1][1].invert_yaxis()

    fig.colorbar(im11, ax=axs[1][1])

    pyplot.setp(axs[-1, :], xlabel='(mm)')
    pyplot.setp(axs[:, 0], ylabel='(mm)')

    pyplot.show()


def show_two_centres(volume, centres, pixel_space):
    y_row = numpy.arange(0.0, volume.shape[0] * pixel_space[0], pixel_space[0])
    x_column = numpy.arange(0.0, volume.shape[1] * pixel_space[1], pixel_space[1])
    z_slice = numpy.arange(0.0, volume.shape[2] * pixel_space[2], pixel_space[2])

    landmark_radius = volume.shape[1] * pixel_space[1] * 0.016

    fig, axs = pyplot.subplots(1, 2, sharex=True)
    fig.set_dpi(300)

    axs[0].set_title("Left Centre")
    axs[0].set_aspect('equal', 'datalim')
    axs[0].pcolormesh(x_column[:], y_row[:], volume[:, :, round(centres[0, 2])], cmap=pyplot.gray())
    llscc_ant = Circle((centres[0, 0] * pixel_space[1], centres[0, 1] * pixel_space[0]), landmark_radius,
                       facecolor='None', edgecolor='r', lw=1)
    axs[0].add_patch(llscc_ant)
    axs[0].invert_yaxis()

    axs[1].set_title("Right Centre")
    axs[1].set_aspect('equal', 'datalim')
    axs[1].pcolormesh(x_column[:], y_row[:], volume[:, :, round(centres[1, 2])], cmap=pyplot.gray())
    llscc_post = Circle((centres[1, 0] * pixel_space[1], centres[1, 1] * pixel_space[0]), landmark_radius,
                        facecolor='None', edgecolor='r', lw=1)
    axs[1].add_patch(llscc_post)
    axs[1].invert_yaxis()

    pyplot.setp(axs[-1], xlabel='(mm)')
    pyplot.setp(axs[0], ylabel='(mm)')

    pyplot.show()


def show_two_centres_cropped(left_volume, right_volume, centres, pixel_space):
    y_row = numpy.arange(0.0, left_volume.shape[0] * pixel_space[0], pixel_space[0])
    x_column = numpy.arange(0.0, left_volume.shape[1] * pixel_space[1], pixel_space[1])
    z_slice = numpy.arange(0.0, left_volume.shape[2] * pixel_space[2], pixel_space[2])

    landmark_radius = left_volume.shape[1] * pixel_space[1] * 0.016

    fig, axs = pyplot.subplots(1, 2, sharex=True)
    fig.set_dpi(300)

    axs[0].set_title("Left Centre")
    axs[0].set_aspect('equal', 'datalim')
    axs[0].pcolormesh(x_column[:], y_row[:], left_volume[:, :, round(centres[0, 2])], cmap=pyplot.gray())
    llscc_ant = Circle((centres[0, 0] * pixel_space[1], centres[0, 1] * pixel_space[0]), landmark_radius,
                       facecolor='None', edgecolor='r', lw=1)
    axs[0].add_patch(llscc_ant)
    axs[0].invert_yaxis()

    axs[1].set_title("Right Centre")
    axs[1].set_aspect('equal', 'datalim')
    axs[1].pcolormesh(x_column[:], y_row[:], right_volume[:, :, round(centres[1, 2])], cmap=pyplot.gray())
    llscc_post = Circle((centres[1, 0] * pixel_space[1], centres[1, 1] * pixel_space[0]), landmark_radius,
                        facecolor='None', edgecolor='r', lw=1)
    axs[1].add_patch(llscc_post)
    axs[1].invert_yaxis()

    pyplot.setp(axs[-1], xlabel='(mm)')
    pyplot.setp(axs[0], ylabel='(mm)')

    pyplot.show()

