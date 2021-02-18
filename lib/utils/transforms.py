# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import cv2

class ImageTransform(object):

    def __init__(self, part):
        self.part = part
        
        
def circle_cutout(image, x_center, y_center, radius, color):
    h, w = image.shape[:-1]
    #print('image shape', image.shape)
    if x_center >= 0 and x_center < image.shape[1] and y_center >= 0 and y_center < image.shape[0]: #switch 1 and 0
        x_center = max(x_center, 0)
        y_center = max(y_center, 0)
        x, y = np.ogrid[-y_center:h - y_center, -x_center:w - x_center]
        mask = x * x + y * y <= radius * radius
        #print('mask', mask)

        image[mask] = color
    return image


def sq_cutout(image, x_center, y_center, side, color):
    #print(image.shape[:2])
    h, w = image.shape[:2] # changing from [:-1] to [:2]
    #print('image shape', image.shape)
    if x_center >= 0 and x_center < image.shape[1] and y_center >= 0 and y_center < image.shape[0]: #switch 1 and 0
        x_center = max(x_center, 0)
        y_center = max(y_center, 0)
        x, y = np.ogrid[-y_center:h - y_center, -x_center:w - x_center]
        #mask = x * x + y * y <= radius * radius
        mask = abs(x + y) + abs(x - y) <= side
        #print('mask', mask)
        image[mask] = color
    return image, x_center, y_center


def cutout_img_custom(img, x, y, mean_coloring, dataset, width = 100):
    color = [0, 0, 0]
    if mean_coloring:
        color = get_image_mean_color(img)

    if x != None and y != None:
        img, x_center, y_center = sq_cutout(img, x, y, width, color)
        return img, (x, y), (width, width), x_center, y_center
    else:
        return img, (0, 0), (0, 0), None, None
    
    
class CutoutCustom(ImageTransform):

    def __init__(self, dataset, mean_coloring=False, width=40):
        #super().__init__(part)
        self.mean_coloring = mean_coloring
        self.dataset = dataset
        self.width = width

    def __call__(self, image, x_c, y_c):
        img = image
        x, y = x_c, y_c
        img, center_pos, widths, x_center, y_center = cutout_img_custom(img, x, y, mean_coloring=self.mean_coloring, dataset = self.dataset, width=self.width)
        return img


def flip_back(output_flipped, matched_parts):
    '''
    ouput_flipped: numpy.ndarray(batch_size, num_joints, height, width)
    '''
    assert output_flipped.ndim == 4,\
        'output_flipped should be [batch_size, num_joints, height, width]'

    output_flipped = output_flipped[:, :, :, ::-1]

    for pair in matched_parts:
        tmp = output_flipped[:, pair[0], :, :].copy()
        output_flipped[:, pair[0], :, :] = output_flipped[:, pair[1], :, :]
        output_flipped[:, pair[1], :, :] = tmp

    return output_flipped


def fliplr_joints(joints, joints_vis, width, matched_parts):
    """
    flip coords
    """
    # Flip horizontal
    joints[:, 0] = width - joints[:, 0] - 1

    # Change left-right parts
    for pair in matched_parts:
        joints[pair[0], :], joints[pair[1], :] = \
            joints[pair[1], :], joints[pair[0], :].copy()
        joints_vis[pair[0], :], joints_vis[pair[1], :] = \
            joints_vis[pair[1], :], joints_vis[pair[0], :].copy()

    return joints*joints_vis, joints_vis


def transform_preds(coords, center, scale, output_size):
    target_coords = np.zeros(coords.shape)
    trans = get_affine_transform(center, scale, 0, output_size, inv=1)
    for p in range(coords.shape[0]):
        target_coords[p, 0:2] = affine_transform(coords[p, 0:2], trans)
    return target_coords


def get_affine_transform(center,
                         scale,
                         rot,
                         output_size,
                         shift=np.array([0, 0], dtype=np.float32),
                         inv=0):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        print(scale)
        scale = np.array([scale, scale])

    scale_tmp = scale * 200.0
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.]).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]


def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result


def crop(img, center, scale, output_size, rot=0):
    trans = get_affine_transform(center, scale, rot, output_size)

    dst_img = cv2.warpAffine(img,
                             trans,
                             (int(output_size[0]), int(output_size[1])),
                             flags=cv2.INTER_LINEAR)

    return dst_img
