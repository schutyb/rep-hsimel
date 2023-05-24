import numpy as np
import tifffile
import hsipy
import matplotlib.pyplot as plt
from skimage.filters import median
from skimage.exposure import equalize_adapthist
from matplotlib import colors
import phasorlibrary as phlib

"""
PRIMERA PARTE 
generate the ome.tiff file compute the phasor, filter and threshold
"""
primero = True
if primero:
    names = ['nevo_15317_07x04']
    tiledim = [[4, 7]]

    # names = ['mel_12635_12x05', 'mel_15116_05x04', 'mel_15719_04x04', 'mel_16400_07x07', 'mel_18852_06x06',
    # 'nevo_15317_07x04', 'nevo_15361_04x02', 'nevo_15397_02x06', 'nevo_16254_05x02', 'nevo_16256_04x02']
    # tiledim = [[5, 12], [4, 5], [4, 4], [7, 7], [6, 6], [4, 7], [2, 4], [6, 2], [2, 5], [2, 4]]

    data = []  # guardo los phasors de los tres casos aca
    for i in range(len(names)):
        im = tifffile.imread('/home/bruno/Documentos/Proyectos/hsimel/Paper/data/spectral/lsm/'
                             + names[i] + '.lsm')
        phasor = np.asarray(hsipy.hsitools.tilephasor(im, dimx=1024, dimy=1024))

        # Stitching
        aux = []
        for j in range(5):
            aux.append(hsipy.hsitools.tile_stitching(phasor[j], tiledim[i][0], tiledim[i][1],
                                                     hper=0.06, vper=0.06, bidirectional=False))
        aux = np.asarray(aux)
        aux2 = np.copy(aux)
        # Filtro con la mediana G y S para limpiar el phasor
        for k in range(5):
            aux2[1] = median(aux2[1])
            aux2[2] = median(aux2[2])
            aux2[3] = median(aux2[3])
            aux2[4] = median(aux2[4])

        aux2[1] = np.where(aux2[0] > 5, aux2[1], aux2[0] * np.nan)
        aux2[2] = np.where(aux2[0] > 5, aux2[2], aux2[0] * np.nan)
        aux2[3] = np.where(aux2[0] > 5, aux2[3], aux2[0] * np.nan)
        aux2[4] = np.where(aux2[0] > 5, aux2[4], aux2[0] * np.nan)

        rgb = phlib.colored_image(aux2[4], np.asarray([45, 180]), outlier_cut=True, color_scale=0.95)

        dcn, gn, sn, mdn, phn = aux2[0], aux2[1], aux2[2], aux2[3], aux2[4]

        plt.figure(1)
        plt.imshow(equalize_adapthist(dcn / dcn.max()), cmap='gray')
        plt.axis('off')
        plt.title('Nevo')

        # Grafico los phasors
        fig1, ax1 = plt.subplots(figsize=(5, 5))
        ax1.hist2d(gn.flatten(), sn.flatten(), bins=256, cmap="RdYlGn_r",
                   norm=colors.LogNorm(), range=[[-1, 1], [-1, 1]])
        phlib.add_color_wheel(ax1)
        phlib.phasor_circle(ax1)

        # Ploteo la imagen de pseudocolor
        plt.figure(5)
        plt.imshow(rgb)
        plt.axis('off')
        plt.show()
