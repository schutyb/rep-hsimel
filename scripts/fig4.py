import numpy as np
import tifffile
import phasorlibrary as phlib
import matplotlib.pyplot as plt
from skimage.filters import median
import os
from sklearn.cluster import KMeans
from hsipy import hsitools
from matplotlib import colors

primero = True
if primero:
    names = ['Elastin', 'FAD+', 'Melanin', 'NADH', 'Porfirin Y']
    for k in range(len(names)):
        im = tifffile.imread('/home/bruno/Documentos/Proyectos/hsimel/datos/componentes/sp/' +
                             names[k] + '.lsm')
        dc, g, s, _, _ = hsitools.phasor(im)
        for j in range(5):
            g = median(g)
            s = median(s)

        # Umbralizar para sacar el background
        g2 = np.where(dc > 5, g, np.zeros(g.shape))
        s2 = np.where(dc > 5, s, np.zeros(g.shape))
        phlib.generate_file('/home/bruno/Documentos/Proyectos/hsimel/datos/componentes/phasors/' +
                            names[k] + '.ome.tiff', np.asarray([dc, g2, s2]))

plot = True
if plot:
    # Grafico los phasors de los componentes
    c = ['Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
         'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
         'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']
    cols = ['grey', 'purple', 'b', 'g', 'orange', 'r',
            'y', 'v', 'k']
    names = ['Elastin', 'FAD+', 'Melanin', 'NADH', 'Porfirin Y']

    plt.figure(figsize=(10, 8))
    ax1 = plt.subplot(2, 2, 1)
    ax2 = plt.subplot(2, 2, 2)
    ax3 = plt.subplot(2, 1, 2)
    axes = [ax1, ax2, ax3]

    for k in range(len(names)):
        im = tifffile.imread('/home/bruno/Documentos/Proyectos/hsimel/datos/componentes/phasors/' +
                             names[k] + '.ome.tiff')

        x, y = hsitools.histogram_thresholding(im[0], im[1], im[2], 5)
        phlib.phasor_circle(ax1)
        ax1.hist2d(x, y, cmap=c[k], bins=256, norm=colors.LogNorm(), range=[[-1, 1], [-1, 1]])

        # Grafico los centroides de los phasors de los componentes
        X1 = np.zeros([2, len(x)])
        X1[0:, 0:] = x, y
        X = X1.T
        cluster = KMeans(n_clusters=1).fit(X)
        coordx, coordy = cluster.cluster_centers_[0][0], cluster.cluster_centers_[0][1]

        circle1 = plt.Circle((coordx, coordy), 0.02, color=cols[k])
        phlib.phasor_circle(ax2)
        ax2.add_patch(circle1)
        ax2.set_xlim(-1, 1)
        ax2.set_ylim(-1, 1)

        # Grafico los espectros promedios de cada componente
        spectrum = []
        lsm = tifffile.imread('/home/bruno/Documentos/Proyectos/hsimel/datos/componentes/sp/' + names[k]
                              + '.lsm')
        aux2 = np.where(im[1] * im[2] != 0, np.ones(lsm[0].shape), np.zeros(lsm[0].shape))
        aux3 = lsm * aux2
        aux4 = np.mean(aux3, axis=0)
        acum = 0
        cont = 0
        for j in range(aux4.shape[0]):
            for k2 in range(aux4.shape[1]):
                if aux4[j][k2] != 0:
                    acum = aux3[:, j, k2] + acum
                    cont = cont + 1
        spectrum = acum / cont

        ax3.plot(np.linspace(418, 718, 30), spectrum / max(spectrum), label=names[k],
                 color=cols[k])
        ax3.legend()
        ax3.set_xlabel('Lambda [nm]')
        ax3.grid()
    plt.show()
