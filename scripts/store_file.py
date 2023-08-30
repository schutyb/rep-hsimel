import numpy as np
import tifffile
from tifffile import imwrite, memmap


def phasor(image_stack, harmonic=1):
    """
        This function computes the average intensity image, the G and S coordinates of the phasor.
    As well as the modulation and phase.

    :param image_stack: is a file with spectral mxm images to calculate the fast fourier transform from
    numpy library.
    :param harmonic: int. The number of the harmonic where the phasor is calculated.
    :return: avg: is the average intensity image
    :return: g: is mxm image with the real part of the fft.
    :return: s: is mxm imaginary with the real part of the fft.
    :return: md: numpy.ndarray  It is the modulus obtain with Euclidean Distance.
    :return: ph: is the phase between g and s in degrees.
    """

    data = np.fft.fft(image_stack, axis=0)

    dc = data[0].real
    dc = np.where(dc != 0, dc, int(np.mean(dc)))  # change the zeros to the img average

    g = data[harmonic].real
    g /= dc
    s = data[harmonic].imag
    s /= -dc

    md = np.sqrt(g ** 2 + s ** 2)
    ph = np.angle(data[harmonic], deg=True)
    avg = np.mean(image_stack, axis=0)

    return avg, g, s, md, ph


def generate_file(filename, gsa):
    """
    :param filename: Type string characters. The name of the file to be written, with the extension ome.tiff
    :param gsa: Type n-dimensional array holding the data to be stored.
    :return file: The file storing the data. If the filename extension was ome.tiff the file is an
    ome.tiff format.
    """
    imwrite(filename, data=gsa)
    file = memmap(filename)
    file.flush()
    return file


im = tifffile.imread('SP_paramesium_561_y_633_R3.lsm')
gsa = phasor(im)
generate_file('test.ome.tiff', gsa)
