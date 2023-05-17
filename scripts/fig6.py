import numpy as np
from hsipy import hsitools, hsi_visualization
import tifffile
import matplotlib.pyplot as plt
from skimage.filters import median
import phasorlibrary as phlib
from matplotlib import colors
import os
from sklearn.cluster import KMeans


names = os.listdir('/home/bruno/Documentos/Proyectos/hsimel/experimentos/rois/phasors/')

fig, ax = plt.subplots(1)
phlib.phasor_circle(ax)
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)

for i in range(10):
    im = tifffile.imread('/home/bruno/Documentos/Proyectos/hsimel/experimentos/rois/phasors/' + names[i])

    plotty = False
    if plotty:
        fig, ax = plt.subplots(1)
        ax.hist2d(im[1].flatten(), im[2].flatten(), bins=256, cmap="RdYlGn_r",
                  norm=colors.LogNorm(), range=[[-1, 1], [-1, 1]])
        phlib.phasor_circle(ax)
        ax.set_title(names[i])
        plt.savefig('/home/bruno/Documentos/Proyectos/hsimel/experimentos/rois/' + names[i] + '.png')

    # Grafico los centroides de los phasors de los componentes
    x = np.delete(im[1].flatten(), np.isnan(im[1]).flatten())
    y = np.delete(im[2].flatten(), np.isnan(im[2]).flatten())
    X1 = np.zeros([2, len(x)])
    X1[0:, 0:] = x, y
    X = X1.T
    cluster = KMeans(n_clusters=1).fit(X)
    coordx, coordy = cluster.cluster_centers_[0][0], cluster.cluster_centers_[0][1]

    if names[i][0:5] == 'roi_m':
        c = 'r'
    else:
        c = 'b'
    circle1 = plt.Circle((coordx, coordy), 0.02, color=c)
    ax.add_patch(circle1)

plt.show()
