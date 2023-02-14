import numpy as np
import tifffile
import phasorlibrary as phlib
import matplotlib.pyplot as plt
from skimage.filters import median
import os
import pandas as pd


compute = False
if compute:
    names = os.listdir('/home/bruno/Documentos/Proyectos/hsimel/datos/componentes/sp/')
    for i in range(len(names)):
        im = tifffile.imread('/home/bruno/Documentos/Proyectos/hsimel/datos/componentes/sp/' + names[i])
        aux = np.asarray(phlib.phasor(im))
        # Filtro con la mediana G y S para limpiar el phasor
        for k in range(4):
            aux[1] = median(aux[1])
            aux[2] = median(aux[2])
            aux[3] = median(aux[3])
            aux[4] = median(aux[4])

        aux[1] = np.where(aux[0] > 1, aux[1], np.zeros(aux[1].shape))
        aux[2] = np.where(aux[0] > 1, aux[2], np.zeros(aux[2].shape))
        aux[3] = np.where(aux[0] > 1, aux[3], np.zeros(aux[3].shape))
        aux[4] = np.where(aux[0] > 1, aux[4], np.zeros(aux[4].shape))

        phlib.generate_file('/home/bruno/Documentos/Proyectos/hsimel/datos/componentes/phasors/' + names[i][:10] +
                            '.ome.tiff', aux)

        plotty = True
        if plotty:
            plt.figure(1)
            phlib.phasor_plot(aux[0], aux[1], aux[2], title=names[i])
            plt.title(names[i])
            plt.savefig('/home/bruno/Documentos/Proyectos/hsimel/datos/componentes/phasors/' + names[i][:10] + '.png')


