
""" some util functions borrowed """
import SimpleITK as sitk
import numpy as np
import math


def rotate_coord(origin, point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in degrees.
    """
    rad = math.radians(angle)

    ox, oy = origin
    px, py = point

    qx = ox + math.cos(rad) * (px - ox) - math.sin(rad) * (py - oy)
    qy = oy + math.sin(rad) * (px - ox) + math.cos(rad) * (py - oy)
    return (qx, qy)


def rotate_3d_coord(origin, point, angle, axes):
    """calculate coord after rotation

    Args:
        origin ([tuple]): [description]
        point ([tuple]): [description]
        angle ([int]): [description]
        axes ([tuple of two]): [see scipy.ndimage.rotate axes description]
    """
    axes = sorted(list(axes))
    origin_2d = tuple(origin[i] for i in axes)
    point_2d = tuple(point[i] for i in axes)
    coord = rotate_coord(origin_2d, point_2d, angle)
    new_point = [i for i in point]
    for i, j in enumerate(axes):
        new_point[j] = coord[i]
    # becase point is coord, so int
    return tuple(int(i) for i in new_point)

# This function is from https://github.com/rock-learning/pytransform3d/blob/7589e083a50597a75b12d745ebacaa7cc056cfbd/pytransform3d/rotations.py#L302


def matrix_from_axis_angle(a):
    """ Compute rotation matrix from axis-angle.
    This is called exponential map or Rodrigues' formula.
    Parameters
    ----------
    a : array-like, shape (4,)
        Axis of rotation and rotation angle: (x, y, z, angle)
    Returns
    -------
    R : array-like, shape (3, 3)
        Rotation matrix
    """
    ux, uy, uz, theta = a
    c = np.cos(theta)
    s = np.sin(theta)
    ci = 1.0 - c
    R = np.array([[ci * ux * ux + c,
                   ci * ux * uy - uz * s,
                   ci * ux * uz + uy * s],
                  [ci * uy * ux + uz * s,
                   ci * uy * uy + c,
                   ci * uy * uz - ux * s],
                  [ci * uz * ux - uy * s,
                   ci * uz * uy + ux * s,
                   ci * uz * uz + c],
                  ])

    # This is equivalent to
    # R = (np.eye(3) * np.cos(theta) +
    #      (1.0 - np.cos(theta)) * a[:3, np.newaxis].dot(a[np.newaxis, :3]) +
    #      cross_product_matrix(a[:3]) * np.sin(theta))

    return R


def resample(image, transform):
    """
    This function resamples (updates) an image using a specified transform
    :param image: The sitk image we are trying to transform
    :param transform: An sitk transform (ex. resizing, rotation, etc.
    :return: The transformed sitk image
    """
    reference_image = image
    interpolator = sitk.sitkLinear
    default_value = 0
    return sitk.Resample(image, reference_image, transform,
                         interpolator, default_value)


def get_center_np(img):
    """ numpy image """
    return [i // 2 for i in img.shape]


def get_center(img):
    """
    This function returns the physical center point of a 3d sitk image
    :param img: The sitk image we are trying to find the center of
    :return: The physical center point of the image
    """
    width, height, depth = img.GetSize()
    return img.TransformIndexToPhysicalPoint((int(np.ceil(width/2)),
                                              int(np.ceil(height/2)),
                                              int(np.ceil(depth/2))))


def rotation3d(image, theta, axis):
    """
    This function rotates an image across each of the x, y, z axes by theta_x, theta_y, and theta_z degrees
    respectively
    :param image: An sitk image
    :param theta: The amount of degrees the user wants the image rotated around an axis
    :param axis: (X, Y, Z)
    :return: The rotated image
    """
    theta = np.deg2rad(theta)
    euler_transform = sitk.Euler3DTransform()
    print(euler_transform.GetMatrix())
    image_center = get_center(image)
    euler_transform.SetCenter(image_center)

    direction = image.GetDirection()
    axis_angle = (direction[0+axis],
                  direction[3+axis],
                  direction[6+axis],
                  theta)
    np_rot_mat = matrix_from_axis_angle(axis_angle)
    euler_transform.SetMatrix(np_rot_mat.flatten().tolist())
    resampled_image = resample(image, euler_transform)
    return resampled_image


def convert_pos(image_size, pos, theta, axis):
    """convert mid point position after rotation
    NOTE: maybe low efficiency, because need another rotation operation
    Args:
        image_size ([tuple]): [size of the image, used to generate sitk image, get from numpy image]
        pos ([tuple]): [mid point position before rotation]
        theta ([int]): [amount of degrees]
        axis ([int]): [which axis to rotate on]
    """
    ref_img = sitk.Image(
        image_size[2], image_size[0], image_size[1], sitk.sitkInt8)
    ref_img[pos[2], pos[0], pos[1]] = 1
    result_img = rotation3d(ref_img, theta, axis=axis)
    coord = _get_coord(np.where(result_img == 1))  # channel-first format
    return (coord[1], coord[2], coord[0])


def _get_coord(array):
    """ get coord tuple from tuple of numpy array """
    return (i.item() for i in array)

