#!/usr/bin/env python
# coding: utf-8

import itertools

import numpy as np

import skimage as ski


rng = np.random.default_rng()


def get_3d_images():
    "Some 3D images"
    cell_4d = ski.data.cells3d()
    return (cell_4d[:20, 0, ::2, ::2],  # Membranes
            cell_4d[:20, 1, ::2, ::2],  # Nuclei
            ski.data.brain())

def rolled_proc(img, axes, func):
    r_img = np.transpose(img, axes)
    f_r_img = func(r_img)
    return np.transpose(f_r_img, np.argsort(axes))


def assert_labels_equivalent(label_1, label_2):
    uq_labels_1 = np.unique(label_1)
    uq_labels_2 = np.unique(label_2)
    assert np.all(uq_labels_1 == uq_labels_2)
    unclaimed = list(uq_labels_2)
    for label in uq_labels_1:
        mask = label_1 == label
        in_mask = label_2[mask]
        label_other = in_mask[0]
        assert np.all(in_mask == label_other)
        unclaimed.remove(label_other)
    assert len(unclaimed) == 0


orderings = set(itertools.permutations(range(3), 3))
orderings.remove((0, 1, 2))


def assert_all_orders(imgs, func, chk_func=assert_labels_equivalent):
    for i, img in enumerate(imgs):
        orig = func(img)
        print(f'Image {i}')
        for order in orderings:
            print(f'Ordering {order}')
            rolled = rolled_proc(img, order, func)
            chk_func(rolled, orig)


def without_ties(img):
    img = ski.util.img_as_float(img)
    noise = rng.normal(0, 0.001, size=img.shape)
    out = img + noise
    assert len(np.unique(out)) == img.size
    return out
