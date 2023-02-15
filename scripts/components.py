import numpy as np
import tifffile
import phasorlibrary as phlib
import matplotlib.pyplot as plt
from skimage.filters import median
import os
from sklearn.cluster import KMeans


compute = False
if compute:
    im = tifffile.imread('/home/bruno/Documentos/Proyectos/hsimel/datos/componentes/20_Porfirin Y.lsm')
    aux = np.asarray(phlib.phasor(im))

    filt = True
    if filt:
        for k in range(5):
            aux[1] = median(aux[1])
            aux[2] = median(aux[2])
        aux[1] = np.where(aux[0] > 2, aux[1], np.zeros(aux[1].shape))
        aux[2] = np.where(aux[0] > 2, aux[2], np.zeros(aux[2].shape))

    phlib.phasor_plot(aux[0], aux[1], aux[2], title='20_Porfirin Y')
    plt.show()

    plotty = True
    if plotty:
        plt.figure(1)
        phlib.phasor_plot(aux[0], aux[1], aux[2], title='20_Porfirin Y')
        plt.title('20_Porfirin Y')
        plt.savefig('/home/bruno/Documentos/Proyectos/hsimel/datos/componentes/20_Porfirin Y.png')
        phlib.generate_file('/home/bruno/Documentos/Proyectos/hsimel/datos/componentes/20_Porfirin Y.ome.tiff', aux)


centros = True
if centros:
    im = tifffile.imread('/home/bruno/Documentos/Proyectos/hsimel/datos/rois/rois2/phasors/mel/melp_001.ome.tiff')

    X1 = np.zeros([2, len(np.concatenate(im[1]))])
    X1[0:, 0:] = np.concatenate(im[1]), np.concatenate(im[2])
    X = X1.T
    # X = X2[~np.isnan(X2).any(axis=1)]

    y = KMeans(n_clusters=1).fit_predict(X)

    plot_cluster = True
    if plot_cluster:
        plt.figure(1)
        plt.scatter(X[:, 0], X[:, 1], c=y)
        plt.title("Phasor clustering")
        plt.xlabel('G')
        plt.ylabel('S')
        plt.show()
