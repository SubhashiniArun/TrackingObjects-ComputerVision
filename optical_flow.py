"""
This file adapted from Sean Batzel's submission to Homework 2.
"""

import argparse
import logging

import imageio
import numpy as np
from scipy.ndimage.filters import convolve


def bilinear_interp(image, points):
    points = np.asarray(points)
    if points.ndim == 1:
        points = points[np.newaxis]

    valid = np.all(points < [image.shape[0] - 1, image.shape[1] - 1], axis=-1)
    valid *= np.all(points >= 0, axis=-1)
    valid = valid.astype(np.float32)
    points = np.minimum(points, [image.shape[0] - 2, image.shape[1] - 2])
    points = np.maximum(points, 0)

    fpart, ipart = np.modf(points)
    tl = ipart.astype(np.int32)
    br = tl + 1
    tr = np.concatenate([tl[..., 0:1], br[..., 1:2]], axis=-1)
    bl = np.concatenate([br[..., 0:1], tl[..., 1:2]], axis=-1)

    b = fpart[..., 0:1]
    a = fpart[..., 1:2]

    top = (1 - a) * image[tl[..., 0], tl[..., 1]] + \
        a * image[tr[..., 0], tr[..., 1]]
    bot = (1 - a) * image[bl[..., 0], bl[..., 1]] + \
        a * image[br[..., 0], br[..., 1]]
    return ((1 - b) * top + b * bot) * valid[..., np.newaxis]


def translate(image, displacement):
    pts = np.mgrid[:image.shape[0], :image.shape[1]
                   ].transpose(1, 2, 0).astype(np.float32)
    pts -= displacement[::-1]

    return bilinear_interp(image, pts)


def convolve_img(image, kernel):
    if kernel.ndim == image.ndim:
        if image.shape[-1] == kernel.shape[-1]:
            return np.dstack([convolve(image[..., c], kernel[..., c]) for c in range(kernel.shape[-1])])
        elif image.ndim == 2:
            return convolve(image, kernel)
        else:
            raise RuntimeError("Invalid kernel shape. Kernel: %s Image: %s" % (
                kernel.shape, image.shape))
    elif kernel.ndim == image.ndim - 1:
        return np.dstack([convolve(image[..., c], kernel) for c in range(image.shape[-1])])
    else:
        raise RuntimeError("Invalid kernel shape. Kernel: %s Image: %s" % (
            kernel.shape, image.shape))


def gaussian_kernel(ksize=5):
    kernel = np.exp(-np.linspace(-(ksize // 2), ksize // 2,
                                 ksize) ** 2 / 2) / np.sqrt(2 * np.pi)
    kernel = np.outer(kernel, kernel)
    kernel /= kernel.sum()
    return kernel


def lucas_kanade(H, I):
    mask = (H.mean(-1) > 0.25) * (I.mean(-1) > 0.25)
    mask = mask[:, :, np.newaxis]

    kernel_x = np.array([[1., 0., -1.],
                         [2., 0., -2.],
                         [1., 0., -1.]]) / 8.

    kernel_y = np.array([[1., 2., 1.],
                         [0., 0., 0.],
                         [-1., -2., -1.]]) / 8.

    I_x = convolve_img(I, kernel_x)
    I_y = convolve_img(I, kernel_y)
    I_t = I - H

    Ixx = (I_x * I_x) * mask
    Ixy = (I_x * I_y) * mask
    Iyy = (I_y * I_y) * mask
    Ixt = (I_x * I_t) * mask
    Iyt = (I_y * I_t) * mask

    AtA = np.array([[Ixx.sum(), Ixy.sum()],
                    [Ixy.sum(), Iyy.sum()]])
                    
    eig_vals, eig_vecs = np.linalg.eig(AtA)

    AtA = np.array([[eig_vals[0], 0],
                    [0, eig_vals[1]]])
                    
    Atb = -np.array([Ixt.sum(), Iyt.sum()])

    displacement = np.linalg.solve(AtA, Atb)

    return displacement, AtA, Atb


def iterative_lucas_kanade(H, I, steps):
    disp = np.zeros((2,), np.float32)
    for i in range(steps):
        tranlated_H = translate(H, disp)

        disp += lucas_kanade(tranlated_H, I)[0]

    return disp


def gaussian_pyramid(image, levels):
    kernel = gaussian_kernel()

    pyr = [image]
    for level in range(1, int(levels)):
        convolved = convolve_img(pyr[level - 1], kernel)

        decimated = convolved[::2, ::2]

        pyr.append(decimated)

    return pyr


def pyramid_lucas_kanade(H, I, initial_d, levels, steps):
    initial_d = np.asarray(initial_d, dtype=np.float32)

    pyramid_H = gaussian_pyramid(H, levels)
    pyramid_I = gaussian_pyramid(I, levels)

    disp = initial_d / 2. ** levels
    for level in range(int(levels)):
        disp *= 2

        level_H = pyramid_H[-(1 + level)]
        level_I = pyramid_I[-(1 + level)]

        level_I_displaced = translate(level_I, -disp)
        disp += iterative_lucas_kanade(level_H, level_I_displaced, steps)

    return disp


def track_object(frame1, frame2, x, y, w, h, steps):
    H = frame1[y:y+h, x:x+w]
    I = frame2[y:y+h, x:x+w]

    levels = np.floor(np.log(w if w < h else h))

    initial_displacement = np.array([0, 0])

    flow = pyramid_lucas_kanade(H, I, initial_displacement, levels, steps)

    final_flow = [0, 0, 0, 0, 0, 0]

    if flow[0] < 0:
        final_flow[0] = abs(flow[0])
    elif flow[0] > 0:
        final_flow[1] = abs(flow[0])

    if flow[1] < 0:
        final_flow[2] = abs(flow[1])
    elif flow[1] > 0:
        final_flow[3] = abs(flow[1])

    return final_flow


def run_lk(first_frame, second_frame, x, y, w, h, steps):
    first = imageio.imread(first_frame)[
        :, :, :3].astype(np.float32) / 255.0
    second = imageio.imread(second_frame)[
        :, :, :3].astype(np.float32) / 255.0
    return track_object(first, second, int(x), int(y), int(w), int(h), steps)


def prepare_dataset(files_list, result_file, x, y, w, h, steps=5):
    import csv
    flow_vector = [0, 0, 0, 0, 0, 0]
    with open(result_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for i in range(len(files_list) - 1):
            flow_vector = run_lk(files_list[i], files_list[i + 1], x + flow_vector[0] +
                                 flow_vector[1], y + flow_vector[2] + flow_vector[3], w, h, steps)
            writer.writerow([files_list[i]])
            writer.writerow([files_list[i + 1]])
            writer.writerow(flow_vector)
