import os

os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2, 40).__str__()
import logging
from typing import Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial
import skimage
import skimage.io
import skimage.morphology
from colorfilters import HSVFilter
from scipy.spatial import distance
from skimage.measure import EllipseModel, label

from hmsm.utils import (
    binarize_image,
    crop_image_to_contents,
    interpolate_missing_pixels,
    morphological_edge_detection,
)

image = cv2.imread("data/animatic/5061372_5_cropped.tif")
footprint = skimage.morphology.diamond(3)

chunk = image[10000:20000, :]

v_channel = (chunk / 255).max(axis=2)

mask_holes = v_channel < 0.10

hole_dillated = skimage.morphology.binary_dilation(mask_holes, footprint)

mask_roll = v_channel > 0.50

mask_line = (np.invert(hole_dillated)) & (v_channel <= 0.75)

mask_line = skimage.morphology.binary_opening(mask_line, footprint)


pixels_cluster = chunk[hole_dillated == False].reshape((-1, 3))
pixels_cluster = np.float32(pixels_cluster)

stop_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 0.2)

_, labels, (centers) = cv2.kmeans(
    pixels_cluster, 2, None, stop_criteria, 10, cv2.KMEANS_RANDOM_CENTERS
)

classes = scipy.spatial.distance.cdist(pixels_cluster, centers)

bg = np.zeros_like(hole_dillated)
bg[hole_dillated == False] = classes

for start in range(0, image.shape[0], 4000):
    end = start + 4000 if start + 4000 < image.shape[0] else image.shape[0]
    chunk = image[start:end, :]
    v_channel = (chunk / 255).max(axis=2)
    mask_line = (v_channel >= 0.1) & (v_channel <= 0.5)
    footprint = skimage.morphology.diamond(3)
    mask_line = skimage.morphology.binary_opening(mask_line, footprint)
    skimage.io.imsave(f"dyn_line_{start}_{end}.jpg", mask_line)

plt.imshow(mask_line)
plt.show()


import json
import statistics

with open("coords.json", "r") as f:
    data = json.load(f)

coords = []

for key, value in data.items():
    coords.append([int(key), int(statistics.mean(value))])

coords = np.vstack(coords)

coords_y = scipy.signal.savgol_filter(coords[:, 1], 200, 3).round()

# def process_roll(path: str):
#     image = cv2.imread("data/5060708_7_cropped.tif", cv2.IMREAD_GRAYSCALE)

#     img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#     color = ("y", "r", "b")
#     plt.figure()
#     for i, col in enumerate(color):
#         histr = cv2.calcHist([img], [i], None, [256], [0, 256])
#         plt.plot(histr, color=col)
#         plt.xlim([0, 256])
#     plt.show()
#     pass
#     cv = chan_vese(
#         img_clr,
#         mu=0.25,
#         lambda1=1,
#         lambda2=1,
#         tol=1e-3,
#         max_num_iter=200,
#         dt=0.5,
#         init_level_set="checkerboard",
#         extended_output=True,
#     )


# import matplotlib.pyplot as plt
# from skimage import color, data, graph, io, morphology, segmentation

# for i, col in enumerate(color):
#     histr = cv2.calcHist([img], [i], None, [256], [0, 256])
#     plt.plot(histr, color=col)
#     plt.xlim([0, 256])
# # Input data
# img = data.immunohistochemistry()

# # Compute a mask
# lum = color.rgb2gray(img)
# mask = morphology.remove_small_holes(
#     morphology.remove_small_objects(lum < 0.7, 500), 500
# )

# mask = morphology.opening(mask, morphology.disk(3))

# # SLIC result
# slic = segmentation.slic(img, n_segments=200, start_label=1)

# # maskSLIC result
# m_slic = segmentation.slic(img, n_segments=100, mask=mask, start_label=1)

# # Display result
# fig, ax_arr = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(10, 10))
# ax1, ax2, ax3, ax4 = ax_arr.ravel()

# ax1.imshow(img)
# ax1.set_title('Original image')

# ax2.imshow(mask, cmap='gray')
# ax2.set_title('Mask')

# ax3.imshow(segmentation.mark_boundaries(img, slic))
# ax3.contour(mask, colors='red', linewidths=1)
# ax3.set_title('SLIC')

# ax4.imshow(segmentation.mark_boundaries(img, m_slic))
# ax4.contour(mask, colors='red', linewidths=1)
# ax4.set_title('maskSLIC')

# for ax in ax_arr.ravel():
#     ax.set_axis_off()

# plt.tight_layout()
# plt.show()

# color = ('h','s','v')
# plt.figure()
# for i,col in enumerate(color):
#     histr = cv2.calcHist([img],[i],None,[256],[0,256])
#     plt.plot(histr,color = col)
#     plt.xlim([0,256])
# plt.show()
