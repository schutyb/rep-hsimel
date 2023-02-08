import numpy as np
import tifffile
import phasorlibrary as ph
import matplotlib.pyplot as plt
from skimage.filters import median
import os
from matplotlib.pyplot import figure
from matplotlib import colors


"""
PRIMERO: calculo los phasors
"""
primero = False
if primero:
    names = os.listdir('/home/bruno/Documentos/Proyectos/hsimel/datos/rois/rois2/sp/')
    for k in range(len(names)):
        im = tifffile.imread('/home/bruno/Documentos/Proyectos/hsimel/datos/rois/rois2/sp/' + names[k])
        dc, g, s, md, phase = ph.phasor(im)
        for j in range(3):
            g = median(g)
            s = median(s)
            md = median(md)
            phase = median(phase)
            print(j)

        # Umbralizar para sacar el background
        g = np.where(dc > 1, g, np.zeros(g.shape))
        s = np.where(dc > 1, s, np.zeros(g.shape))
        md = np.where(dc > 1, md, np.zeros(g.shape))
        phase = np.where(dc > 1, phase, np.zeros(g.shape))
        ph.generate_file('/home/bruno/Documentos/Proyectos/hsimel/datos/rois/rois2/phasors/' +
                         names[k][0:8] + '.ome.tiff', np.asarray([dc, g, s, md, phase]))

"""
SEGUNDO: calculo los histogramas de mod y phase
"""
segundo = False
if segundo:
    binsph = np.arange(45, 180)
    binsmd = np.linspace(0, 1, 100)
    names = os.listdir('/home/bruno/Documentos/Proyectos/hsimel/datos/rois/rois2/phasors/')
    hmd = []
    hph = []
    stdmd = []
    stdph = []
    for i in range(3):
        histmd = []
        histph = []
        for k in range(i*10, i*10+10):

            im = tifffile.imread('/home/bruno/Documentos/Proyectos/hsimel/datos/rois/rois2/phasors/' + names[k])
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

    plotty = True
    if plotty:
        for i in range(3):
            plt.plot(binsph[0:134], hph[i] / max(hph[i]), 'ko-')
            plt.fill_between(binsph[0:134], hph[i] / max(hph[i]) - stdph[i] / max(stdph[i]),
                             hph[i] / max(hph[i]) + stdph[i] / max(stdph[i]))
            plt.yscale('log')
            plt.show()


tercero = True
if tercero:
    names = os.listdir('/home/bruno/Documentos/Proyectos/hsimel/datos/rois/rois2/phasors/')
    for i in range(len(names)):
        im = tifffile.imread('/home/bruno/Documentos/Proyectos/hsimel/datos/rois/rois2/phasors/' + names[i])
        rgb = ph.colored_image(im[4], np.asarray([45, 180]), outlier_cut=False, color_scale=1)
        histph = np.histogram(np.concatenate(im[4]), bins=np.arange(45, 180))[0]
        histmd = np.histogram(np.concatenate(im[3]), bins=np.arange(0, 1, 0.01))[0]

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 15))
        fig.suptitle('Nevus/Melanoma fluorescence profile')
        ax1.hist2d(np.concatenate(im[1]), np.concatenate(im[2]), bins=256, cmap="RdYlGn_r",
                   norm=colors.LogNorm(), range=[[-1, 1], [-1, 1]])
        ph.phasor_circle(ax1)
        ax2.imshow(rgb)
        ax2.axis('off')

        ax2.set_title('Pseudocolor image')
        ax1.set_title('Phasor')
        ax3.set_title('Phase histogram')
        ax4.set_title('Modulation histogram')

        ax3.plot(np.arange(45, 179), histph/max(histph))
        ax4.plot(np.arange(0, 1, 0.01)[:len(histmd)], histmd/max(histmd))
        plt.savefig('/home/bruno/Documentos/Proyectos/hsimel/datos/rois/rois2/pseudocolor/'
                    + names[i][:10] + '.png', bbox_inches='tight')
