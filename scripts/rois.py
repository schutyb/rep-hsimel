import numpy as np
import tifffile
import phasorlibrary as ph
import matplotlib.pyplot as plt
from skimage.filters import median
import os
from matplotlib import colors
from hsipy import hsitools

"""
PRIMERO: calculo los phasors
"""
primero = False
if primero:
    names = os.listdir('/home/bruno/Documentos/Proyectos/hsimel/datos/rois/regiones/nev/spectral/')
    for k in range(len(names)):
        im = tifffile.imread('/home/bruno/Documentos/Proyectos/hsimel/datos/rois/regiones/nev/spectral/' +
                             names[k])  # sumar + names en k
        dc, g, s, md, phase = hsitools.phasor(im)
        for j in range(3):
            g = median(g)
            s = median(s)
            md = median(md)
            phase = median(phase)

        # Umbralizar para sacar el background
        g2 = np.where(dc > 1, g, np.zeros(g.shape))
        s2 = np.where(dc > 1, s, np.zeros(g.shape))
        md2 = np.where(dc > 1, md, np.zeros(g.shape))
        phase2 = np.where(dc > 1, phase, np.zeros(g.shape))
        # color = ph.colored_image(phase, phinterval=[45, 180])

        ph.generate_file('/home/bruno/Documentos/Proyectos/hsimel/datos/rois/regiones/nev/phasors/' +
                         names[k][0:12] + '.ome.tiff', np.asarray([dc, g2, s2, md2, phase2]))

"""
SEGUNDO: calculo los histogramas de mod y phase con la std
"""
segundo = False
if segundo:
    binsph = np.arange(45, 180)
    binsmd = np.linspace(0, 1, 100)
    hmd = []
    hph = []
    stdmd = []
    stdph = []
    names = os.listdir('/home/bruno/Documentos/Proyectos/hsimel/datos/rois/regiones/nev/phasors/')
    histmd = []
    histph = []
    for k in range(len(names)):
        im = tifffile.imread('/home/bruno/Documentos/Proyectos/hsimel/datos/rois/regiones/nev/phasors/' +
                             names[k])
        histmd.append(np.histogram(np.concatenate(im[3]), bins=binsmd)[0])
        histph.append(np.histogram(np.concatenate(im[4]), bins=binsph)[0])
        aux_hmd = np.mean(np.asarray(histmd), axis=0)
        aux_hph = np.mean(np.asarray(histph), axis=0)
        aux_stdmd = np.std(np.asarray(histmd), axis=0)
        aux_stdph = np.std(np.asarray(histph), axis=0)
        hmd.append(aux_hmd)
        hph.append(aux_hph)
        stdmd.append(aux_stdmd)
        stdph.append(aux_stdph)
    hmd = np.asarray(hmd)
    hph = np.asarray(hph)
    stdmd = np.asarray(stdmd)
    stdph = np.asarray(stdph)

"""
CUARTO: calculo el centro de masa de la distribucion de cada histograma de los 10 de cada grupo
"""
cuarto = True
if cuarto:
    binsph = np.arange(45, 180)
    binsmd = np.linspace(0, 1, 100)
    tipo = ['mel', 'nev']
    c = ['r', 'g']

    for i in range(len(tipo)):
        names = os.listdir('/home/bruno/Documentos/Proyectos/hsimel/datos/rois/regiones/' + tipo[i] + '/phasors')
        hist = False
        if hist:
            histmd = np.zeros(len(binsmd) - 2)
            histph = np.zeros(len(binsph) - 2)
            for k in range(len(names)):
                im = tifffile.imread('/home/bruno/Documentos/Proyectos/hsimel/datos/rois/regiones/' + tipo[i] +
                                     '/phasors/' + names[k])
                histmd = np.histogram(np.concatenate(im[3]), bins=binsmd)[0][1:] + histmd
                histph = np.histogram(np.concatenate(im[4]), bins=binsph)[0][1:] + histph

                gc, sc = hsitools.histogram_thresholding(im[0], im[1], im[2], 1)
                imcolored = hsitools.phase_modulation_image(im[4], np.asarray([45, 180]))

                fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(30, 8))
                ax1.hist2d(gc, sc, bins=256, cmap="RdYlGn_r", norm=colors.LogNorm(), range=[[-1, 1], [-1, 1]])
                ph.phasor_circle(ax1)
                ax2.imshow(imcolored)
                ax3.plot(histph, 'r')
                ax4.plot(histmd, 'b')
                plt.savefig('/home/bruno/Documentos/Proyectos/hsimel/datos/rois/regiones/imagenes/' + tipo[i] + '_'
                            + names[k][0:12] + '.png', bbox_inches='tight')
                plt.close()

        cm = True  # calculo el centro de masa
        if cm:
            imd = np.zeros([2, len(names)])
            iph = np.zeros([2, len(names)])
            for k in range(len(names)):
                im = tifffile.imread('/home/bruno/Documentos/Proyectos/hsimel/datos/rois/regiones/' + tipo[i] +
                                     '/phasors/' + names[k])
                histmd = np.histogram(np.concatenate(im[3]), bins=binsmd)[0][1:]
                histph = np.histogram(np.concatenate(im[4]), bins=binsph)[0][1:]
                #  calculo el centro de masa de cada histograma y lo guardo en una lista
                imd[i][k] = ph.center_of_mass(binsmd[:98], histmd)
                iph[i][k] = ph.center_of_mass(binsph[:133], histph)

            plt.figure(1)
            plt.plot(i * np.ones(len(iph[i])), iph[i], 'o', color=c[i], label=tipo[i])
            plt.legend()
            plt.ylabel('Phase')
            plt.grid()

            plt.figure(2)
            plt.plot(i * np.ones(len(imd[i])), imd[i], 'o', color=c[i], label=tipo[i])
            plt.legend()
            plt.ylabel('Modulation')
            plt.grid()

            plt.figure(3)
            plt.plot(imd[i], iph[i], 'o', color=c[i], label=tipo[i])
            plt.xlabel('Modulation')
            plt.ylabel('Phase')
            plt.legend()
            plt.grid()
    plt.show()
