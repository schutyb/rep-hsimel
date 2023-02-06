import numpy as np
import tifffile
import phasorlibrary as ph
import matplotlib.pyplot as plt
from skimage.filters import median
import os
from skimage.exposure import equalize_adapthist
from matplotlib.pyplot import figure


"""
PRIMERO grafico los histogramas de las rois
"""
primero = False
if primero:
    tipo = ['nevo', 'nevopig', 'melanomas']
    binsph = np.arange(45, 180)
    binsmd = np.linspace(0, 1, 135)
    for i in range(3):
        names = os.listdir('/home/bruno/Documentos/Proyectos/hsimel/datos/rois/' + tipo[i] + '/phasors/')
        plt.figure(i)
        for k in range(len(names)):
            im = tifffile.imread('/home/bruno/Documentos/Proyectos/hsimel/datos/rois/' + tipo[i] + '/phasors/' +
                                 names[k])
            histph = np.histogram(np.concatenate(im[4]), bins=binsph)
            histmd = np.histogram(np.concatenate(im[3]), bins=binsmd)

            plotty = True
            if plotty:
                plt.plot(histmd[0][1:] / max(histmd[0][1:]), histph[0][1:] / max(histph[0][1:]), label=names[k][:20])
                plt.legend()
                # plt.yscale('log')
        plt.savefig('/home/bruno/Documentos/Proyectos/hsimel/datos/rois/' + tipo[i] + '/' + '/pseudocolor/'
                    + 'df' + names[k][:20] + '.png', bbox_inches='tight')

"""
Estudio estadistico de regiones
"""
segundo = True
if segundo:
    # Melanoma region de la lesion
