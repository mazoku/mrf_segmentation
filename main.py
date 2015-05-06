__author__ = 'tomas'

import skimage.data as skidat
import matplotlib.pyplot as plt
import numpy as np

import MarkovRandomField

#- - - - - - - - - - - - - - - - - - -

# loading the image
img = skidat.camera()
n_rows, n_cols = img.shape

# seed points
seeds_1 = (np.array([90, 252, 394]), np.array([220, 212, 108]))  # first class
seeds_2 = (np.array([68, 265, 490]), np.array([92, 493, 242]))  # second class
seeds = np.zeros((n_rows, n_cols), dtype=np.uint8)
seeds[seeds_1] = 1
seeds[seeds_2] = 2

# plt.figure()
# plt.imshow(seeds, interpolation='nearest')
# plt.show()

# plt.figure()
# plt.imshow(img, 'gray')
# plt.show()

# run(img, seeds)

scale = 0.5  # scaling parameter for resizing the image
alpha = 1  # parameter for weighting the smoothness term (pairwise potentials)
beta = 1  # parameter for weighting the data term (unary potentials)
mrf = MarkovRandomField.MarkovRandomField(img, seeds, alpha=alpha, beta=beta, scale=scale)
mrf.run()