from __future__ import division

import numpy as np
import scipy.stats as scista

class ColorModel():

    def __init__(self, mean, sigma, type='pdf', max_val=None):
        # self.x = x
        self.mean = mean
        self.sigma = sigma
        self.rv = scista.norm(self.mean, self.sigma)
        if type == 'pdf':
            self.prob_func = scista.norm(self.mean, self.sigma).pdf
        elif type == 'cdf':
            self.prob_func = scista.norm(self.mean, self.sigma).cdf
        elif type == 'sf':
            self.prob_func = scista.norm(self.mean, self.sigma).sf
        self.max_val = max_val

    def get_val(self, x):
        y = self.prob_func(x)
        if self.max_val is not None:
            y = y / y.max() * self.max_val
        return y

    def get_pdf(self, x):
        y =self.rv.pdf(x)
        if self.max_val is not None:
            y = y / y.max() * self.max_val
        return y

    def get_cdf(self, x):
        y =self.rv.cdf(x)
        if self.max_val is not None:
            y = y / y.max() * self.max_val
        return y

    def get_sf(self, x):
        y =self.rv.sf(x)
        if self.max_val is not None:
            y = y / y.max() * self.max_val
        return y

    def get_log_val(self, x):
        y = self.get_val(x)
        y = - np.log(y)
        return y