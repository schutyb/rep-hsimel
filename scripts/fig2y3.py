"""
PRIMERA PARTE: calculo y ploteo el phasor de los tres casos de las imagenes completas
SEGUNDA PARTE: calculo y ploteo el phasor de las ROI's
TERCERA PARTE: Pseudocolor
CUARTA PARTE: Histogramas
"""

import numpy as np
import tifffile
import phasorlibrary as ph
import matplotlib.pyplot as plt
from skimage.filters import median
import os
from skimage.exposure import equalize_adapthist
from matplotlib import colors

"""
PRIMERA PARTE 
"""
primero = False
if primero:
    # names = ['SP_16256_6x3_bidir_gain600_4avg', 'SP_15237_9x8_bidir_gain600_4avg']
    names = ['16256_SP_7x3_bidir_autofocus_gain600']

    # tiledim = [[3, 6], [8, 9], [10, 10]]
    tiledim = [[3, 7]]
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

    plotty = True
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

            # Umbralizar para sacar el background
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
            im = tifffile.imread('/home/bruno/Documentos/Proyectos/hsimel/datos/rois/' + tipo[i] + '/phasors/' +
                                 names[k])
            plot_phase = True  # ploteo la imagen de la fase con una escala espectral para ver como se ve
            if plot_phase:
                rgb = ph.colored_image(im[4], np.asarray([45, 180]), outlier_cut=False, color_scale=1)
                ph.generate_file('/home/bruno/Documentos/Proyectos/hsimel/datos/rois/' + tipo[i] + '/pseudocolor/' +
                                 names[k][0:20] + '.ome.tiff', rgb)
                plt.imshow(rgb)
                plt.savefig('/home/bruno/Documentos/Proyectos/hsimel/datos/rois/' + tipo[i] + '/' + '/pseudocolor/'
                            + names[k][:20] + '.png', bbox_inches='tight')

plott = False
if plott:
    # Imagen de Pseudocolor
    im = tifffile.imread('/home/bruno/Documentos/Proyectos/hsimel/datos/16256_SP_7x3_bidir_autofocus_gain600.ome.tiff')
    rgb = ph.colored_image(im[4], np.asarray([45, 180]), outlier_cut=False, color_scale=1)
    plt.imshow(rgb)
    plt.axis('off')
    plt.show()
    plt.imshow(equalize_adapthist(im[0] / im[0].max()), cmap='gray')
    ph.phasor_plot(im[0], im[1], im[2])

"""
CUARTA PARTE grafico los hitogramas de las tres imagenes completas
"""
cuarto = False
if cuarto:
    imn = tifffile.imread('/home/bruno/Documentos/Proyectos/hsimel/datos/completas/'
                          '16256_SP_7x3_bidir_autofocus_gain600.ome.tiff')
    imnp = tifffile.imread('/home/bruno/Documentos/Proyectos/hsimel/datos/completas/'
                           'SP_15237_9x8_bidir_gain600_4avg.ome.tiff')
    imm = tifffile.imread('/home/bruno/Documentos/Proyectos/hsimel/datos/completas/'
                          '18852_SP_bidir_9x10_autofocus_gain600.ome.tiff')

    bins = np.arange(45, 180)
    binsmd = np.linspace(0, 1, 101)
    # hago el histograma de las fases
    histn = np.histogram(np.concatenate(imn[3]), bins=binsmd)
    histnp = np.histogram(np.concatenate(imnp[3]), bins=binsmd)
    histm = np.histogram(np.concatenate(imm[3]), bins=binsmd)

    histnph = np.histogram(np.concatenate(imn[4]), bins=bins)
    histnpph = np.histogram(np.concatenate(imnp[4]), bins=bins)
    histmph = np.histogram(np.concatenate(imm[4]), bins=bins)

    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.plot(binsmd[1:100], histn[0][1:] / max(histn[0][1:]), 'g', label='Nevus')
    # ax1.plot(binsmd[1:100], histnp[0][1:] / max(histnp[0][1:]), 'b', label='Pigmented nevus')
    ax1.plot(binsmd[1:100], histm[0][1:] / max(histm[0][1:]), 'b', label='Melanoma')
    ax1.set_yscale('log')
    ax1.set_xlabel('Modulation')
    ax1.grid()
    # ax1.legend()

    ax2.plot(bins[0:134], histnph[0] / max(histnph[0]), 'g', label='Nevus')
    # ax2.plot(bins[0:134], histnpph[0] / max(histnpph[0]), 'b', label='Pigmented nevus')
    ax2.plot(bins[0:134], histmph[0] / max(histmph[0]), 'b', label='Melanoma')
    ax2.set_yscale('log')
    ax2.set_xlabel('Phase')
    ax2.grid()
    # ax2.legend()

    plt.show()

"Ploteo el phasor con el spectro para mostrar la escala"
quinto = False
if quinto:
    _, gn, sn, _, _ = tifffile.imread('/home/bruno/Documentos/Proyectos/hsimel/datos/completas/'
                                      '16256_SP_7x3_bidir_autofocus_gain600.ome.tiff')
    _, gm, sm, _, _ = tifffile.imread('/home/bruno/Documentos/Proyectos/hsimel/datos/completas/'
                                      '18852_SP_bidir_9x10_autofocus_gain600.ome.tiff')

    auxn = gn * sn
    gn = np.delete(gn.flatten(), np.where(auxn.flatten() == 0))
    sn = np.delete(sn.flatten(), np.where(auxn.flatten() == 0))

    auxm = gm * sm
    gm = np.delete(gm.flatten(), np.where(auxm.flatten() == 0))
    sm = np.delete(sm.flatten(), np.where(auxm.flatten() == 0))

    fig1, ax1 = plt.subplots(figsize=(10, 10))
    ax1.hist2d(gn, sn, bins=256, cmap="RdYlGn_r", norm=colors.LogNorm(), range=[[-1, 1], [-1, 1]])
    ph.add_color_wheel(ax1)
    ph.phasor_circle(ax1)

    fig2, ax2 = plt.subplots(figsize=(10, 10))
    ax2.hist2d(gm, sm, bins=256, cmap="RdYlGn_r", norm=colors.LogNorm(), range=[[-1, 1], [-1, 1]])
    ph.phasor_circle(ax2)
    ph.add_color_wheel(ax2)

    plt.show()

