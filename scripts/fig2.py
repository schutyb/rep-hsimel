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
primero = False
if primero:
    names = ['nevo_16256_04x02', 'mel_18852_06x06']
    tiledim = [[2, 4], [6, 6]]

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

        phlib.generate_file('/home/bruno/Documentos/Proyectos/hsimel/Paper/data/phasors/' + names[i] +
                            '.ome.tiff', aux2)

"""
Segunda parte, calculo y grafico la imagen de pseudocolor
la guardo en un tiff
uso sola la fase para la pseudoclor con rango de 45 a 180
"""
segundo = False
if segundo:
    dc, g, s, md, ph = tifffile.imread('/home/bruno/Documentos/Proyectos/hsimel/Paper/figures/fig2y3/'
                                       'nevo_16256_04x02.ome.tiff')
    rgb = phlib.colored_image(ph, np.asarray([45, 180]), outlier_cut=True, color_scale=0.95)
    # abajo guardo el tiff del rgb
    phlib.generate_file('/home/bruno/Documentos/Proyectos/hsimel/Paper/figures/fig2y3/rgb16256', rgb)

"""
Gráficas las imagenes de dc, phasor con la escala espectral 
y las imagenes de pseudoclor
"""
plotty = True
if plotty:
    dcn, gn, sn, mdn, phn = tifffile.imread('/home/bruno/Documentos/Proyectos/hsimel/Paper/figures/fig2y3/'
                                            'nevo_16256_04x02.ome.tiff')
    dcm, gm, sm, mdm, phm = tifffile.imread('/home/bruno/Documentos/Proyectos/hsimel/Paper/figures/fig2y3/'
                                            'mel_18852_06x06.ome.tiff')
    plt.figure(1)
    plt.imshow(equalize_adapthist(dcn / dcn.max()), cmap='gray')
    plt.axis('off')
    plt.title('Nevo')

    plt.figure(2)
    plt.imshow(equalize_adapthist(dcm / dcm.max()), cmap='gray')
    plt.axis('off')
    plt.title('Melanoma')

    # Grafico los phasors
    fig1, ax1 = plt.subplots(figsize=(5, 5))
    ax1.hist2d(gn.flatten(), sn.flatten(), bins=256, cmap="RdYlGn_r",
               norm=colors.LogNorm(), range=[[-1, 1], [-1, 1]])
    phlib.add_color_wheel(ax1)
    phlib.phasor_circle(ax1)

    fig2, ax2 = plt.subplots(figsize=(5, 5))
    ax2.hist2d(gm.flatten(), sm.flatten(), bins=256, cmap="RdYlGn_r",
               norm=colors.LogNorm(), range=[[-1, 1], [-1, 1]])
    phlib.add_color_wheel(ax2)
    phlib.phasor_circle(ax2)

    # Ploteo la imagen de pseudocolor
    rgbn = tifffile.imread('/home/bruno/Documentos/Proyectos/hsimel/Paper/figures/fig2y3/rgb16256')
    plt.figure(5)
    plt.imshow(rgbn)
    plt.axis('off')

    rgbm = tifffile.imread('/home/bruno/Documentos/Proyectos/hsimel/Paper/figures/fig2y3/rgb18852')
    plt.figure(6)
    plt.imshow(rgbm)
    plt.axis('off')
    plt.show()
