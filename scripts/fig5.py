# La primera parte calcula los phasors de los componentes y los almacena en un tiff
# La segunda parte los grafica, al phasor, luego el CM calculado con clustering
# y luego grafica el espectro promedio.

import numpy as np
import tifffile
import phasorlibrary as phlib
import matplotlib.pyplot as plt
from skimage.filters import median
from sklearn.cluster import KMeans
from hsipy import hsitools
from matplotlib import colors

primero = False
if primero:
    names = ['Colageno_II', 'Colageno_IV', 'Colageno_III', 'Colageno_I', 'NADH',
             'Elastin', 'Melanin', 'FAD+']
    for k in range(len(names)):
        im = tifffile.imread('/home/bruno/Documentos/Proyectos/hsimel/Paper/data/components/' +
                             names[k] + '.lsm')
        dc, g, s, _, _ = hsitools.phasor(im)
        for j in range(5):
            g = median(g)
            s = median(s)

        # Umbralizar para sacar el background
        g2 = np.where(dc > 5, g, g*np.nan)
        s2 = np.where(dc > 5, s, s*np.nan)
        phlib.generate_file('/home/bruno/Documentos/Proyectos/hsimel/Paper/data/fig5/phasors/' +
                            names[k] + '.ome.tiff', np.asarray([dc, g2, s2]))

plot = True
if plot:
    # Grafico los phasors de los componentes
    c = ['Reds', 'Oranges', 'YlOrBr', 'Greens', 'YlOrRd', 'BuGn', 'Blues', 'Purples']
    cols = ['r', 'orange', 'yellow', 'chartreuse', 'olivedrab', 'cyan', 'b', 'purple']
    names = ['Colageno_II', 'Colageno_IV', 'Colageno_III', 'Colageno_I', 'NADH',
             'Elastin', 'Melanin', 'FAD+']

    fig1, ax1 = plt.subplots(1)
    fig2, ax2 = plt.subplots(1)
    fig3, ax3 = plt.subplots(1, figsize=(8, 5))
    for k in range(len(names)):
        im = tifffile.imread('/home/bruno/Documentos/Proyectos/hsimel/Paper/figures/fig5/phasors/' +
                             names[k] + '.ome.tiff')

        imaux = im[1] * im[2]
        x = np.delete(im[1].flatten(), np.isnan(imaux).flatten())
        y = np.delete(im[2].flatten(), np.isnan(imaux).flatten())

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
        lsm = tifffile.imread('/home/bruno/Documentos/Proyectos/hsimel/Paper/data/components/' + names[k]
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
        ax3.set_ylabel('Normalized intensity')
        ax3.grid()
    plt.show()
