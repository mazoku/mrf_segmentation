__author__ = 'tomas'

import skimage.data as skidat
import matplotlib.pyplot as plt


def run():
    print 'this is the main script'


if __name__ == '__main__':
    img = skidat.camera()

    plt.figure()
    plt.imshow(img, 'gray')
    plt.show()