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

from color_model import ColorModel


class MarkovRandomField:

    def __init__(self, img, seeds=None, n_objects=2, mask=None, alpha=1, beta=1, scale=0, models_estim=None, verbose=True):
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
        if seeds is not None:
            self.seeds = seeds.copy()  # working variable - i.e. resized data etc.
        else:
            self.seeds = None
        if mask is None:
            self.mask_orig = np.ones_like(self.img)
        else:
            self.mask_orig = mask
        self.mask = self.mask_orig.copy()

        self.n_slices, self.n_rows, self.n_cols = self.img_orig.shape
        if seeds is not None:
            self.n_seeds = (seeds > 0).sum()
        else:
            self.n_seeds = 0

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

        self.models = None  # list of intensity models used for segmentation

        self.verbose = verbose

        if models_estim is None:
            if seeds is not None:
                self.models_estim = 'seeds'
            else:
                self.models_estim = 'n_objects'
        elif models_estim == 'hydohy':
            self.n_objects = 3
            self.models_estim = models_estim

        self.params = self.load_parameters()
        params = {'alpha': alpha, 'beta': beta, 'scale': scale}
        self.params.update(params)

    def load_parameters(self, config_path='config.ini'):
        # load parameters
        params_default = {
            'win_l': 50,
            'win_w': 350,
            'alpha': 4,
            'beta': 1,
            'zoom': 0,
            'scale': 0.25,
            'perc': 30,
            'k_std_dom': 5,
            'k_std_hypo': 1,
            'domin_simple_estim': 0,
            'prob_w': 0.001,
            'unaries_as_cdf': 0,
            'bgd_label': 0,
            'hypo_label': 1,
            'domin_label': 2,
            'hyper_label': 3,
            'hypo_mean_offset': -20,
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

        # self.params.update(self.load_parameters())
        params_default.update(params)

        return params_default

    def _debug(self, msg, lineend):
        if self.verbose:
            if lineend:
                print msg
            else:
                print msg,

    def estimate_dominant_pdf(self):
        perc = self.params['perc']
        k_std = self.params['k_std_dom']
        simple_estim = self.params['domin_simple_estim']

        ints = self.img[np.nonzero(self.mask)]
        hist, bins = skiexp.histogram(ints, nbins=256)
        if simple_estim:
            mu, sigma = scista.norm.fit(ints)
        else:
            n_pts = self.mask.sum()
            perc_in = n_pts * perc / 100

            peak_idx = np.argmax(hist)
            n_in = hist[peak_idx]
            win_width = 0

            while n_in < perc_in:
                win_width += 1
                n_in = hist[peak_idx - win_width:peak_idx + win_width].sum()
                self._debug('win_w=%i, perc_in=%.0f, n_in=%.0f' % (win_width, perc_in, n_in), True)

            start_id = max(0, peak_idx - win_width)
            idx_start = bins[start_id]
            end_id = min(peak_idx + win_width, len(bins) - 1)
            idx_end = bins[end_id]
            inners_m = np.logical_and(ints > idx_start, ints < idx_end)

            inners = ints[np.nonzero(inners_m)]

            mu = bins[peak_idx]
            sigma = k_std * np.std(inners)

            # muse, sigmase = scista.norm.fit(ints)
            # print 'simple -> (%.1f, %.2f)' % (muse, sigmase)
            # print '  perc -> (%.1f, %.2f)' % (mu, sigma)

        # mu = int(mu)
        # sigma = int(round(sigma))
        cm = ColorModel(mu, sigma, type='pdf')
        return cm

    def estimate_outlier_pdf(self, rv_domin, outlier_type):
        self._debug('estimate_outlier_pdf: %s' % outlier_type, True)
        prob_w = self.params['prob_w']

        probs = rv_domin.get_val(self.img) * self.mask

        max_prob = rv_domin.get_val(rv_domin.mean)

        prob_t = prob_w * max_prob

        ints_out_m = (probs < prob_t) * self.mask

        ints_out = self.img[np.nonzero(ints_out_m)]

        if outlier_type == 'hypo':
            ints = ints_out[np.nonzero(ints_out < rv_domin.mean)]
            if ints.size == 0:
                mu = 0
                sigma = 1
            else:
                mu, sigma = scista.norm.fit(ints)
            if self.params['unaries_as_cdf']:
                cm = ColorModel(mu + self.params['hypo_mean_offset'], sigma * self.params['k_std_hypo'], type='sf', max_val=max_prob)
                # cm = ColorModel(mu + self.params['hypo_mean_offset'], sigma * self.params['k_std_hypo'], type='sf')
            else:
                cm = ColorModel(mu + self.params['hypo_mean_offset'], sigma, type='pdf', max_val=max_prob)
                # cm = ColorModel(mu + self.params['hypo_mean_offset'], sigma, type='pdf')
            # y1 = scista.beta(1, 4).pdf(x)
        elif outlier_type == 'hyper':
            ints = ints_out[np.nonzero(ints_out > rv_domin.mean)]
            if ints.size == 0:
                mu = 255
                sigma = 1
            else:
                mu, sigma = scista.norm.fit(ints)
            if self.params['unaries_as_cdf']:
                cm = ColorModel(mu, sigma, type='cdf', max_val=max_prob)
                # cm = ColorModel(mu, sigma, type='cdf')
            else:
                cm = ColorModel(mu, sigma, type='pdf', max_val=max_prob)
                # cm = ColorModel(mu, sigma, type='pdf')
        else:
            print 'Wrong outlier specification.'
            return

        return cm

    def calc_models(self):
        if self.models_estim == 'seeds':
            models = self.calc_models_seeds()
        elif self.models_estim == 'hydohy':
            models = self.calc_models_hydohy()
        else:
            raise ValueError('Wrong type of model estimation mode.')

        return models

    def calc_models_seeds(self):
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

    def calc_models_hydohy(self):
        # print 'calculating intensity models...'
        # dominant class ------------
        rv_domin = self.estimate_dominant_pdf()
        self._debug('\tdominant pdf: mu = %.1f, sigma = %.1f' % (rv_domin.mean, rv_domin.sigma), True)

        # hypodense class ------------
        rv_hypo = self.estimate_outlier_pdf(rv_domin, 'hypo')
        self._debug('\thypo pdf: mu = %.1f, sigma = %.1f' % (rv_hypo.mean, rv_hypo.sigma), True)

        # hyperdense class ------------
        rv_hyper = self.estimate_outlier_pdf(rv_domin, 'hyper')
        self._debug('\thyper pdf: mu = %.1f, sigma = %.1f' % (rv_hyper.mean,  rv_hyper.sigma), True)

        models = [rv_hypo, rv_domin, rv_hyper]

        return models

    def plot_models(self, nbins=256, show_now=True):
        plt.figure()
        x = np.arange(self.img.min(), self.img.max())  # artificial x-axis

        hist, bins = skiexp.histogram(self.img, nbins=nbins)
        plt.plot(bins, hist, 'k')
        plt.hold(True)
        if self.models is not None:
            probs = [cm.get_val(x) for cm in self.models]
            y_max = max([p.max() for p in probs])
            fac = hist.max() / y_max

            colors = 'rgbcmy' * 10
            for i, p in enumerate(probs):
                plt.plot(x, fac * p, colors[i], linewidth=2)
            if show_now:
                plt.show()

    def get_unaries(self, ret_prob=False, show=False, show_now=True):
        if self.models is None:
            self.models = self.calc_models()

        # unaries_l = [x.get_log_val(self.img) for x in self.models]
        unaries_l = [skiexp.rescale_intensity(x.get_inverse(self.img), out_range=np.uint8) for x in self.models]

        if show:
            plt.figure()
            for i, u in enumerate(unaries_l):
                plt.subplot(1, 3, i + 1)
                plt.imshow(u.reshape(self.img.shape)[0, :, :], 'gray')
                plt.colorbar()
                plt.title('unary #%i' % (i + 1))

            if show_now:
                plt.show()

        un_probs_l = [x.get_val(self.img) for x in self.models]

        unaries = np.dstack((x.reshape(-1, 1) for x in unaries_l))
        un_probs = np.dstack((x.reshape(-1, 1) for x in un_probs_l))

        # x = np.arange(0, 255, 1)
        # y_hypo_p = self.models[0].get_val(x)
        # y_dom_p = self.models[1].get_val(x)
        # y_hyper_p = self.models[2].get_val(x)
        #
        # # y_hypo_u = self.models[0].get_log_val(x)
        # # y_dom_u = self.models[1].get_log_val(x)
        # # y_hyper_u = self.models[2].get_log_val(x)
        # y_hypo_u = self.models[0].get_inverse(x)
        # y_dom_u = self.models[1].get_inverse(x)
        # y_hyper_u = self.models[2].get_inverse(x)

        # plt.figure()
        # plt.plot(x, y_hypo_p, 'b-')
        # plt.plot(x, y_dom_p, 'g-')
        # plt.plot(x, y_hyper_p, 'r-')
        # plt.xlim([0, 255])
        # plt.title('probabilities (cdf models)')
        #
        # plt.figure()
        # plt.plot(x, y_hypo_u, 'b-')
        # plt.plot(x, y_dom_u, 'g-')
        # plt.plot(x, y_hyper_u, 'r-')
        # plt.xlim([0, 255])
        # plt.title('unary term (cdf models)')

        if ret_prob:
            return unaries.astype(np.int32), un_probs
            # return unaries, un_probs
        else:
            return unaries.astype(np.int32)
            # return unaries

    def set_unaries(self, unaries, resize=False):
        '''
        Set unary term.
        :param unaries: list of unary terms - item per object, item has to be an ndarray
        :param resize: if to resize to match the image (scaled down by factor self.scale) shape ot raise an error
        :return:
        '''
        if (unaries.shape[0] != np.prod(self.img.shape)):
        #     if resize:
        #         unaries = [cv2.resize(x, self.img.shape) for x in unaries]
        #     else:
            raise ValueError('Wrong unaries shape. Either input the right shape (n_pts, 1, n_objs) or allow resizing.')

        # unaries = np.dstack((x.reshape(-1, 1) for x in unaries))

        if self.n_objects is not None and self.n_objects != unaries.shape[-1]:
            self.n_objects = unaries.shape[-1]
            self.set_pairwise()
        self.unaries = unaries

    def set_pairwise(self):
        self.pairwise = - self.alpha * np.eye(self.n_objects, dtype=np.int32)

    def show_slice(self, slice_id, show_now=True):
        plt.figure()
        plt.subplot(221), plt.imshow(self.img_orig[slice_id, :, :], 'gray', interpolation='nearest'), plt.title('input image')
        plt.subplot(222), plt.imshow(self.seeds_orig[slice_id, :, :], interpolation='nearest'), plt.title('seeds')
        plt.subplot(223), plt.imshow(self.labels[slice_id, :, :], interpolation='nearest'), plt.title('segmentation')
        plt.subplot(224), plt.imshow(skiseg.mark_boundaries(self.img[slice_id, :, :].astype(np.uint8), self.labels[slice_id, :, :]),
                                     interpolation='nearest'), plt.title('segmentation')
        if show_now:
            plt.show()

    def run(self, resize=True):
        #----  rescaling  ----
        if resize and self.scale != 0:
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
        # if self.unaries is None:
        if self.models is None:
            self._debug('calculating intensity models ...', False)
            # self.models = self.calc_intensity_models()
            self.models = self.calc_models()
            self._debug('done', True)

        #----  creating unaries  ----
        if self.unaries is None:
            self._debug('calculating unary potentials ...', False)
            self.unaries = self.beta * self.get_unaries()
            self._debug('done', True)

        #----  create potts pairwise  ----
        if self.pairwise is None:
            self._debug('calculating pairwise potentials ...', False)
            # self.pairwise = - self.alpha * np.eye(self.n_objects, dtype=np.int32)
            self.set_pairwise()
            self._debug('done', True)

        #----  deriving graph edges  ----
        self._debug('deriving graph edges ...', False)
        # use the gerneral graph algorithm
        # first, we construct the grid graph
        # inds = np.arange(self.n_rows * self.n_cols).reshape(self.img.shape)
        # horz = np.c_[inds[:, :-1].ravel(), inds[:, 1:].ravel()]
        # vert = np.c_[inds[:-1, :].ravel(), inds[1:, :].ravel()]
        # self.edges = np.vstack([horz, vert]).astype(np.int32)
        inds = np.arange(self.img.size).reshape(self.img.shape)
        if self.img.ndim == 2:
            horz = np.c_[inds[:, :-1].ravel(), inds[:, 1:].ravel()]
            vert = np.c_[inds[:-1, :].ravel(), inds[1:, :].ravel()]
            self.edges = np.vstack([horz, vert]).astype(np.int32)
        elif self.img.ndim == 3:
            horz = np.c_[inds[:, :, :-1].ravel(), inds[:, :, 1:].ravel()]
            vert = np.c_[inds[:, :-1, :].ravel(), inds[:, 1:, :].ravel()]
            dept = np.c_[inds[:-1, :, :].ravel(), inds[1:, :, :].ravel()]
            self.edges = np.vstack([horz, vert, dept]).astype(np.int32)
        # deleting edges with nodes outside the mask
        nodes_in = np.ravel_multi_index(np.nonzero(self.mask), self.img.shape)
        rows_inds = np.in1d(self.edges, nodes_in).reshape(self.edges.shape).sum(axis=1) == 2
        self.edges = self.edges[rows_inds, :]
        self._debug('done', True)

        #----  calculating graph cut  ----
        self._debug('calculating graph cut ...', False)
        # we flatten the unaries
        result_graph = pygco.cut_from_graph(self.edges, self.unaries.reshape(-1, self.n_objects), self.pairwise)
        self.labels = result_graph.reshape(self.img.shape)
        self._debug('done', True)

        #----  zooming to the original size  ----
        if resize and self.scale != 0:
            # self.labels_orig = cv2.resize(self.labels, (0,0),  fx=1. / self.scale, fy= 1. / self.scale, interpolation=cv2.INTER_NEAREST)
            self.labels_orig = tools.resize3D(self.labels, 1. / self.scale, sliceId=0)
        else:
            self.labels_orig = self.labels

        self._debug('----------', True)
        self._debug('segmentation done', True)

        # self.show_slice(0)

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

        return self.labels_orig


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
    unaries = mrf.get_unaries()
    mrf.set_unaries(unaries)

    plt.figure()
    plt.subplot(131), plt.imshow(img, 'gray')
    plt.subplot(132), plt.imshow(unaries[:, :, 0].reshape(img.shape), 'gray', interpolation='nearest')
    plt.subplot(133), plt.imshow(unaries[:, :, 1].reshape(img.shape), 'gray', interpolation='nearest')
    plt.show()

    labels = mrf.run()

    plt.figure()
    plt.subplot(221), plt.imshow(img, 'gray', interpolation='nearest'), plt.title('input image')
    plt.subplot(222), plt.imshow(seeds, interpolation='nearest')
    # plt.hold(True)
    # seeds_v = np.nonzero(self.seeds)
    # for i in range(len(seeds_v[0])):
    #     seed = (seeds_v[0][i], seeds_v[1][i])
    #     if self.seeds[seed]
    #
    # plt.plot
    plt.title('seeds')
    plt.subplot(223), plt.imshow(labels[0, :, :], interpolation='nearest'), plt.title('segmentation')
    plt.subplot(224), plt.imshow(skiseg.mark_boundaries(img, labels[0, :, :]), interpolation='nearest'), plt.title('segmentation')
    plt.show()