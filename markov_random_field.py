__author__ = 'tomas'

import numpy as np
import scipy.stats as scista
import matplotlib.pyplot as plt
import skimage.data as skidat

import skimage.segmentation as skiseg

import pygco

import cv2

import sys
sys.path.   append('../imtools/')
from imtools import tools


class MarkovRandomField:

    def __init__(self, img, seeds=None, mask=None, alpha=1, beta=1, scale=0):
        if img.ndim == 2:
            img = np.expand_dims(img, 0)
        if mask is not None and mask.ndim == 2:
            mask = np.expand_dims(mask, 0)
        if seeds is not None and seeds.ndim == 2:
            seeds = np.expand_dims(seeds, 0)

        self.img_orig = img  # original variable
        # self.img = None  # working variable - i.e. resized data etc.
        self.img = img.copy()  # working variable - i.e. resized data etc.
        self.seeds_orig = seeds  # original variable
        self.seeds = seeds.copy()  # working variable - i.e. resized data etc.
        if mask == None:
            self.mask_orig = np.ones_like(self.img)
        else:
            self.mask_orig = mask
        self.mask = self.mask_orig.copy()

        self.n_slices, self.n_rows, self.n_cols = self.img_orig.shape
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

    def set_unaries(self, unaries, resize=False):
        if (self.img_orig.shape == unaries.shape).all():
            self.unaries = unaries
        elif resize:
            self.unaries = cv2.resize(unaries, self.img_orig.shape)
        else:
            raise ValueError('Wrong unaries shape. Either input the right shape (same as input image) or allow resizing.')

    def show_slice(self, slice_id, show_now=True):
        plt.figure()
        plt.subplot(221), plt.imshow(self.img_orig[slice_id, :, :], 'gray', interpolation='nearest'), plt.title('input image')
        plt.subplot(222), plt.imshow(self.seeds_orig[slice_id, :, :], interpolation='nearest'), plt.title('seeds')
        plt.subplot(223), plt.imshow(self.labels[slice_id, :, :], interpolation='nearest'), plt.title('segmentation')
        plt.subplot(224), plt.imshow(skiseg.mark_boundaries(self.img[slice_id, :, :].astype(np.uint8), self.labels[slice_id, :, :]),
                                     interpolation='nearest'), plt.title('segmentation')
        if show_now:
            plt.show()


    def run(self):
        #----  rescaling  ----
        if self.scale != 0:
            self.img = tools.resize3D(self.img_orig, self.scale, sliceId=0)
            self.seeds = tools.resize3D(self.seeds_orig, self.scale, sliceId=0)
            self.mask = tools.resize3D(self.mask_orig, self.scale, sliceId=0)
            # for i, (im, seeds, mask) in enumerate(zip(self.img_orig, self.seeds_orig, self.mask_orig)):
            #     self.img[i, :, :] = cv2.resize(im, (0,0), fx=self.scale, fy=self.scale, interpolation=cv2.INTER_NEAREST)
            #     self.seeds[i, :, :] = cv2.resize(seeds, (0,0),  fx=self.scale, fy=self.scale, interpolation=cv2.INTER_NEAREST)
            #     self.mask[i, :, :] = cv2.resize(mask, (0,0),  fx=self.scale, fy=self.scale, interpolation=cv2.INTER_NEAREST)
        # else:
        #     self.img = self.img_orig
        #     self.seeds = self.seeds_orig
        self.n_slices, self.n_rows, self.n_cols = self.img.shape

        #----  calculating intensity models  ----
        if self.unaries is None:
            print 'calculating intensity models ...',
            self.models = self.calc_intensity_models()
            print 'done'

        #----  creating unaries  ----
        if self.unaries is None:
            print 'calculating unary potentials ...',
            self.unaries = self.beta * self.get_unaries()
            print 'done'

        #----  create potts pairwise  ----
        if self.pairwise is None:
            print 'calculating pairwise potentials ...',
            self.pairwise = - self.alpha * np.eye(self.n_objects, dtype=np.int32)
            print 'done'

        #----  deriving graph edges  ----
        print 'deriving graph edges ...',
        # use the gerneral graph algorithm
        # first, we construct the grid graph
        # inds = np.arange(self.n_rows * self.n_cols).reshape(self.img.shape)
        # horz = np.c_[inds[:, :-1].ravel(), inds[:, 1:].ravel()]
        # vert = np.c_[inds[:-1, :].ravel(), inds[1:, :].ravel()]
        # self.edges = np.vstack([horz, vert]).astype(np.int32)
        inds = np.arange(self.img.size).reshape(self.img.shape)
        if img.ndim == 2:
            horz = np.c_[inds[:, :-1].ravel(), inds[:, 1:].ravel()]
            vert = np.c_[inds[:-1, :].ravel(), inds[1:, :].ravel()]
            self.edges = np.vstack([horz, vert]).astype(np.int32)
        elif img.ndim == 3:
            horz = np.c_[inds[:, :, :-1].ravel(), inds[:, :, 1:].ravel()]
            vert = np.c_[inds[:, :-1, :].ravel(), inds[:, 1:, :].ravel()]
            dept = np.c_[inds[:-1, :, :].ravel(), inds[1:, :, :].ravel()]
            self.edges = np.vstack([horz, vert, dept]).astype(np.int32)
        # deleting edges with nodes outside the mask
        nodes_in = np.ravel_multi_index(np.nonzero(self.mask), self.img.shape)
        rows_inds = np.in1d(self.edges, nodes_in).reshape(self.edges.shape).sum(axis=1) == 2
        self.edges = self.edges[rows_inds, :]
        print 'done'

        #----  calculating graph cut  ----
        print 'calculating graph cut ...',
        # we flatten the unaries
        result_graph = pygco.cut_from_graph(self.edges, self.unaries.reshape(-1, self.n_objects), self.pairwise)
        self.labels = result_graph.reshape(self.img.shape)
        print 'done'

        #----  zooming to the original size  ----
        if self.scale != 0:
            # self.labels_orig = cv2.resize(self.labels, (0,0),  fx=1. / self.scale, fy= 1. / self.scale, interpolation=cv2.INTER_NEAREST)
            self.labels_orig = tools.resize3D(self.labels, 1. / self.scale, sliceId=0)

        print '----------'
        print 'segmentation done'

        self.show_slice(0)
        # plt.figure()
        # plt.subplot(221), plt.imshow(self.img_orig[0, :, :], 'gray', interpolation='nearest'), plt.title('input image')
        # plt.subplot(222), plt.imshow(self.seeds_orig[0, :, :], interpolation='nearest')
        # # plt.hold(True)
        # # seeds_v = np.nonzero(self.seeds)
        # # for i in range(len(seeds_v[0])):
        # #     seed = (seeds_v[0][i], seeds_v[1][i])
        # #     if self.seeds[seed]
        # #
        # # plt.plot
        # plt.title('seeds')
        # plt.subplot(223), plt.imshow(self.labels, interpolation='nearest'), plt.title('segmentation')
        # plt.subplot(224), plt.imshow(skiseg.mark_boundaries(self.img_orig, self.labels), interpolation='nearest'), plt.title('segmentation')
        # plt.show()


#-----------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------
if __name__ == '__main__':
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
    mrf = MarkovRandomField(img, seeds, alpha=alpha, beta=beta, scale=scale)
    mrf.run()