__author__ = 'tomas'

import sys
sys.path.append('../../imtools/')
from imtools import tools

import numpy as np
import matplotlib.pyplot as plt

import cv2
import skimage.data as skidat
import skimage.segmentation as skiseg
import skimage.exposure as skiexp

import pygco
import scipy.stats as scista
import ConfigParser


class MarkovRandomField:

    def __init__(self, img, seeds=None, n_objects=2, mask=None, alpha=1, beta=1, scale=0, models_estim=None):
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
        if seeds is not None:
            self.n_objects = self.seeds.max()
        else:
            self.n_objects = n_objects

        self.unaries = None  # unary term = data term
        self.pairwise = None  # pairwise term = smoothness term
        self.labels = None  # labels of the final segmentation

        self.models = list()  # list of intensity models used for segmentation

        if models_estim is None:
            if seeds is not None:
                self.models_estim = 'seeds'
            else:
                self.models_estim = 'n_objects'
        elif models_estim == 'hydohy':
            self.n_objects = 3
            self.models_estim = models_estim

    def load_parameters(self, config_path='config.ini'):
        # load parameters
        self.params = {
            'win_l': 50,
            'win_w': 350,
            'alpha': 4,
            'beta': 1,
            'zoom': 0,
            'scale': 0.25,
            'perc': 30,
            'k_std_h': 3,
            'domin_simple_estim': 0,
            'prob_w': 0.0001,
            'unaries_as_cdf': 0,
            'bgd_label': 0,
            'hypo_label': 1,
            'domin_label': 2,
            'hyper_label': 3
            # 'voxel_size': (1, 1, 1)
        }

        config = ConfigParser.ConfigParser()
        config.read(config_path)

        params = dict()

        # an automatic way
        for section in config.sections():
            for option in config.options(section):
                try:
                    params[option] = config.getint(section, option)
                except ValueError:
                    try:
                        params[option] = config.getfloat(section, option)
                    except ValueError:
                        if option == 'voxel_size':
                            str = config.get(section, option)
                            params[option] = np.array(map(int, str.split(', ')))
                        else:
                            params[option] = config.get(section, option)

        self.params.update(self.load_parameters())

    def estimate_dominant_pdf(self):
        perc = self.params['perc']
        k_std_l = self.params['k_std_h']
        simple_estim = self.params['domin_simple_estim']

        ints = self.img[np.nonzero(self.mask)]
        hist, bins = skiexp.histogram(ints, nbins=256)
        if simple_estim:
            mu, sigma = scista.norm.fit(ints)
        else:
            ints = self.img[np.nonzero(self.mask)]

            n_pts = self.mask.sum()
            perc_in = n_pts * perc / 100

            peak_idx = np.argmax(hist)
            n_in = hist[peak_idx]
            win_width = 0

            while n_in < perc_in:
                win_width += 1
                n_in = hist[peak_idx - win_width:peak_idx + win_width].sum()

            idx_start = bins[peak_idx - win_width]
            idx_end = bins[peak_idx + win_width]
            inners_m = np.logical_and(ints > idx_start, ints < idx_end)
            inners = ints[np.nonzero(inners_m)]

            # liver pdf -------------
            mu = bins[peak_idx]
            sigma = k_std_l * np.std(inners)

        mu = int(mu)
        sigma = int(sigma)
        rv = scista.norm(mu, sigma)

        return rv

    def estimate_outlier_pdf(self, rv_domin, outlier_type):
        prob_w = self.params['prob_w']

        probs = rv_domin.pdf(self.img) * self.mask

        max_prob = rv_domin.pdf(rv_domin.mean())

        prob_t = prob_w * max_prob

        ints_out_m = probs < prob_t * self.mask

        ints_out = self.img[np.nonzero(ints_out_m)]

        if outlier_type == 'hypo':
            ints = ints_out[np.nonzero(ints_out < rv_domin.mean())]
        elif outlier_type == 'hyper':
            ints = ints_out[np.nonzero(ints_out > rv_domin.mean())]
        else:
            print 'Wrong outlier specification.'
            return

        mu, sigma = scista.norm.fit(ints)

        mu = int(mu)
        sigma = int(sigma)
        rv = scista.norm(mu, sigma)

        return rv

    def calc_models(self):
        if self.models_estim == 'seeds':
            models = self.calc_seeds_models()
        elif self.models_estim == 'hydohy':
            models = self.calc_hydohy_models()
        else:
            raise ValueError('Wrong type of model estimation mode.')

        return models

    def calc_seeds_models(self):
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

    def calc_hydohy_models(self):
        # print 'calculating intensity models...'
        # dominant class pdf ------------
        rv_domin = self.estimate_dominant_pdf()
        print '\tdominant pdf: mu = ', rv_domin.mean(), ', sigma = ', rv_domin.std()

        # hypodense class pdf ------------
        rv_hypo = self.estimate_outlier_pdf('hypo')
        print '\thypo pdf: mu = ', rv_hypo.mean(), ', sigma = ', rv_hypo.std()

        # hyperdense class pdf ------------
        rv_hyper = self.estimate_outlier_pdf('hyper')
        print '\thyper pdf: mu = ', rv_hyper.mean(), ', sigma = ', rv_hyper.std()

        models = [rv_hypo, rv_domin, rv_hyper]

        return models

    def get_unaries(self):
        unaries_l = [- model.logpdf(self.img) for model in self.models]
        unaries = np.dstack((x.reshape(-1, 1) for x in unaries_l))

        return unaries.astype(np.int32)

    def set_unaries(self, unaries, resize=False):
        '''
        Set unary term.
        :param unaries: list of unary terms - item per object, item has to be an ndarray
        :param resize: if to resize to match the image (scaled down by factor self.scale) shape ot raise an error
        :return:
        '''
        if (unaries[0].shape != self.img.shape).all():
            if resize:
                unaries = [cv2.resize(x, self.img.shape) for x in unaries]
            else:
                raise ValueError('Wrong unaries shape. Either input the right shape (1, n_pts, n_objs) or allow resizing.')

        unaries = np.dstack((x.reshape(-1, 1) for x in unaries))

        self.n_objects = unaries.shape[2]
        self.unaries = unaries

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
            # self.models = self.calc_intensity_models()
            self.models = self.calc_models()
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