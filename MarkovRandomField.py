__author__ = 'tomas'

import numpy as np
import scipy.stats as scista
import matplotlib.pyplot as plt

import skimage.segmentation as skiseg

import pygco

import cv2

class MarkovRandomField:

    def __init__(self, img, seeds, alpha=1, beta=1, scale=0):
        self.img_orig = img  # original variable
        self.img = None  # working variable - i.e. resized data etc.
        self.seeds_orig = seeds  # original variable
        self.seeds = seeds  # working variable - i.e. resized data etc.

        self.n_rows, self.n_cols = self.img_orig.shape
        self.n_seeds = (self.seeds > 0).sum()

        self.alpha = alpha
        self.beta = beta
        self.scale = scale
        self.n_objects = self.seeds.max()

        self.unaries = None  # unary term = data term
        self.pairwise = None  # pairwise term = smoothness term
        self.labels = None  # labels of the final segmentation

        self.models = list()  # list of intensity models used for segmentation


    def calc_intensity_models(self):
        models = list()
        for i in range(1, self.n_objects + 1):
            pts = self.img[np.nonzero(self.seeds == i)]
            mu = np.mean(pts)
            sigma = np.std(pts)

            mu = int(mu)
            sigma = int(sigma)
            rv = scista.norm(mu, sigma)
            models.append(rv)

        return models


    def get_unaries(self):
        unaries = np.zeros((self.n_rows, self.n_cols, self.n_objects))
        for i in range(self.n_objects):
            unaries[:, :, i] = - self.models[i].logpdf(self.img)

        return unaries.astype(np.int32)


    def run(self):
        #----  rescaling  ----
        if self.scale != 0:
            self.img = cv2.resize(self.img_orig, (0,0),  fx=self.scale, fy=self.scale, interpolation=cv2.INTER_NEAREST)
            self.seeds = cv2.resize(self.seeds_orig, (0,0),  fx=self.scale, fy=self.scale, interpolation=cv2.INTER_NEAREST)
        else:
            self.img = self.img_orig
            self.seeds = self.seeds_orig
        self.n_rows, self.n_cols = self.img.shape

        #----  calculating intensity models  ----
        print 'calculating intensity models...'
        self.models = self.calc_intensity_models()

        #----  creating unaries  ----
        print 'calculating unary potentials...'
        self.unaries = self.beta * self.get_unaries()

        #----  create potts pairwise  ----
        print 'calculating pairwise potentials...'
        self.pairwise = - self.alpha * np.eye(self.n_objects, dtype=np.int32)

        #----  deriving graph edges  ----
        print 'deriving graph edges...'
        # use the gerneral graph algorithm
        # first, we construct the grid graph
        inds = np.arange(self.n_rows * self.n_cols).reshape(self.img.shape)
        horz = np.c_[inds[:, :-1].ravel(), inds[:, 1:].ravel()]
        vert = np.c_[inds[:-1, :].ravel(), inds[1:, :].ravel()]
        self.edges = np.vstack([horz, vert]).astype(np.int32)

        #----  calculating graph cut  ----
        print 'calculating graph cut...'
        # we flatten the unaries
        result_graph = pygco.cut_from_graph(self.edges, self.unaries.reshape(-1, self.n_objects), self.pairwise)
        self.labels = result_graph.reshape(self.img.shape)

        #----  zooming to the original size  ----
        if self.scale != 0:
            self.labels = cv2.resize(self.labels, (0,0),  fx=1. / self.scale, fy= 1. / self.scale, interpolation=cv2.INTER_NEAREST)

        print '----------'
        print 'segmentation done'

        plt.figure()
        plt.subplot(221), plt.imshow(self.img_orig, 'gray', interpolation='nearest'), plt.title('input image')
        plt.subplot(222), plt.imshow(self.seeds_orig, interpolation='nearest')
        # plt.hold(True)
        # seeds_v = np.nonzero(self.seeds)
        # for i in range(len(seeds_v[0])):
        #     seed = (seeds_v[0][i], seeds_v[1][i])
        #     if self.seeds[seed]
        #
        # plt.plot
        plt.title('seeds')
        plt.subplot(223), plt.imshow(self.labels, interpolation='nearest'), plt.title('segmentation')
        plt.subplot(224), plt.imshow(skiseg.mark_boundaries(self.img_orig, self.labels), interpolation='nearest'), plt.title('segmentation')
        plt.show()