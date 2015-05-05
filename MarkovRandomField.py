__author__ = 'tomas'

import numpy as np
import scipy.stats as scista

import cv2

class MarkovRandomField:

    def __init__(self, img, seeds, alpha=1, beta=1, scale=0):
        self.img_orig = img  # original variable
        self.img = None  # working variable - i.e. resized data etc.
        self.seeds_orig = seeds  # original variable
        self.seeds = None  # working variable - i.e. resized data etc.

        self.alpha = alpha
        self.beta = beta
        self.scale = scale
        self.n_objects = self.seeds.max()

        self.models = list()  # list of intensity models used for segmentation

        if self.scale != 0:
            self.img = cv2.resize(self.img_orig, (0,0),  fx=scale, fy=self.scale, interpolation=cv2.INTER_NEAREST)
            self.seeds = cv2.resize(self.seeds_orig, (0,0),  fx=scale, fy=self.scale, interpolation=cv2.INTER_NEAREST)
        else:
            self.img = self.img_orig
            self.seeds = self.seeds_orig


    def calc_intensity_models(self):
        models = list()
        for i in range(self.n_objects):
            pts = self.img[np.nonzero(self.seeds)]
            mu = np.mean(pts)
            sigma = np.std(pts)

            mu = int(mu)
            sigma = int(sigma)
            rv = scista.norm(mu, sigma)
            models.append(rv)


    def run(self):
