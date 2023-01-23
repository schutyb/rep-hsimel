"""
PRIMERA PARTE: calculo y ploteo el phasor de los tres casos de las imagenes completas
SEGUNDA PARTE: calculo y ploteo el phasor de las ROI's
TERCERA PARTE: Pseudocolor
"""

import numpy as np
import tifffile
import phasorlibrary as ph
import matplotlib.pyplot as plt
from skimage.filters import median
import os

"""
PRIMERA PARTE 
"""
primero = False
if primero:
    names = ['SP_16256_6x3_bidir_gain600_4avg', 'SP_15237_9x8_bidir_gain600_4avg']
    # 'SP_18852_10x10_bidir_autofocus' volver a hacer tiene desenfoque
    tiledim = [[3, 6], [8, 9], [10, 10]]
    data = []  # guardo los phasors de los tres acasos aca
    for i in range(len(names)):
        im = tifffile.imread('/home/bruno/Documentos/Proyectos/hsimel/datos/' + names[i] + '.lsm')  # Leo los archivos
        phasor = ph.phasor_tile(im, 1024, 1024)  # Calculo el phasor del tile
        # Concateno las 5 imagenes que calcula phasor_tile
        aux = []
        for j in range(5):
            aux.append(ph.concatenate(phasor[j], tiledim[i][0], tiledim[i][1], bidirectional=True, hper=0.07))
        aux = np.asarray(aux)
        # Filtro con la mediana G y S para limpiar el phasor
        for k in range(3):
            aux[1] = median(aux[1])
            aux[2] = median(aux[2])
            aux[3] = median(aux[3])
            aux[4] = median(aux[4])

        aux[1] = np.where(aux[0] > 1, aux[1], np.zeros(aux[1].shape))
        aux[2] = np.where(aux[0] > 1, aux[2], np.zeros(aux[2].shape))
        aux[3] = np.where(aux[0] > 1, aux[3], np.zeros(aux[3].shape))
        aux[4] = np.where(aux[0] > 1, aux[4], np.zeros(aux[4].shape))

        ph.generate_file('/home/bruno/Documentos/Proyectos/hsimel/datos/' + names[i] + '.ome.tiff', aux)
        data.append(aux)

    plotty = False
    if plotty:
        # Grafico las HSI
        # revisar la funcion from skimage.exposure import equalize_adapthist para hacer la hsi
        # da problema porque los valores son entre -1 y 1
        plt.figure(1)
        plt.imshow(data[0][0], cmap='gray')
        plt.title('Nevo sin pigmento')
        plt.figure(2)
        plt.imshow(data[1][0], cmap='gray')
        plt.title('Nevo pigmentado')
        plt.figure(3)
        plt.imshow(data[2][0], cmap='gray')
        plt.title('Melanoma')

        # Grafico los phasors
        fig1 = ph.phasor_plot(np.asarray(data[0][0]), np.asarray(data[0][1]), np.asarray(data[0][2]), np.asarray([3]),
                              title='Nevo sin pigmento')
        fig2 = ph.phasor_plot(np.asarray(data[1][0]), np.asarray(data[1][1]), np.asarray(data[1][2]), np.asarray([3]),
                              title='Nevo pigmentado')
        fig3 = ph.phasor_plot(np.asarray(data[2][0]), np.asarray(data[2][1]), np.asarray(data[2][2]), np.asarray([3]),
                              title='Melanoma')
        plt.show()

"""
SEGUNDA PARTE 
"""

segundo = False
if segundo:
    tipo = ['nevo', 'nevopig', 'melanomas']
    for i in range(3):
        names = os.listdir('/home/bruno/Documentos/Proyectos/hsimel/datos/rois/' + tipo[i] + '/sp/')
        for k in range(len(names)):
            im = tifffile.imread('/home/bruno/Documentos/Proyectos/hsimel/datos/rois/' + tipo[i] + '/sp/' + names[k])
            dc, g, s, md, phase = ph.phasor(im)
            for j in range(3):
                g = median(g)
                s = median(s)
                md = median(md)
                phase = median(phase)

            g = np.where(dc > 1, g, np.zeros(g.shape))
            s = np.where(dc > 1, s, np.zeros(g.shape))
            md = np.where(dc > 1, md, np.zeros(g.shape))
            phase = np.where(dc > 1, phase, np.zeros(g.shape))
            ph.generate_file('/home/bruno/Documentos/Proyectos/hsimel/datos/rois/' + tipo[i] + '/phasors/' +
                             names[k][0:32] + '.ome.tiff', np.asarray([dc, g, s, md, phase]))

            plotty = False
            if plotty:
                fig = ph.phasor_plot(dc, g, s, np.asarray([3]), title=names[k])
                plt.savefig(
                    '/home/bruno/Documentos/Proyectos/hsimel/datos/rois/' + tipo[i] + '/' + 'phasor' + names[k] +
                    '.png', bbox_inches='tight')
                plt.figure(2)
                plt.imshow(dc, cmap='gray')
                plt.axis('off')
                plt.title(names[k])
                plt.savefig('/home/bruno/Documentos/Proyectos/hsimel/datos/rois/' + tipo[i] + '/' + 'hsi' + names[k] +
                            '.png', bbox_inches='tight')

"""
TERCERA PARTE 
"""
tercero = False
if tercero:
    tipo = ['nevo', 'nevopig', 'melanomas']
    for i in range(3):
        names = os.listdir('/home/bruno/Documentos/Proyectos/hsimel/datos/rois/' + tipo[i] + '/phasors/')
        for k in range(len(names)):
            im = tifffile.imread('/home/bruno/Documentos/Proyectos/hsimel/datos/rois/' + tipo[i] + '/phasors/' + names[k])

            plot_phase = False  # ploteo la imagen de la fase con una escala espectral para ver como se ve
            if plot_phase:
                plt.imshow(im[4], cmap='nipy_spectral')
                plt.savefig('/home/bruno/Documentos/Proyectos/hsimel/datos/rois/' + tipo[i] + names[k] +
                            '.png', bbox_inches='tight')

# Imagen de Pseudocolor
im = tifffile.imread('/home/bruno/Documentos/Proyectos/hsimel/datos/SP_15237_9x8_bidir_gain600_4avg.ome.tiff')
rgb = ph.colored_image(im[4], np.asarray([45, 180]),
                       outlier_cut=False, color_scale=1)
plt.imshow(rgb)
plt.show()
