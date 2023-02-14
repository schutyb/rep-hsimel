import numpy as np
import tifffile
import phasorlibrary as phlib
import matplotlib.pyplot as plt
from skimage.filters import median
import os
import pandas as pd

"""
PRIMERA PARTE 
"""
primero = False
if primero:
    binsph = np.arange(45, 180)
    binsmd = np.linspace(0, 1, 101)
    names = os.listdir('/home/bruno/Documentos/Proyectos/hsimel/experimentos/melanomas/sp/')
    data = []  # guardo los phasors de los tres acasos aca
    imd = np.zeros(len(names))
    iph = np.zeros(len(names))
    for i in range(len(names)):
        im = tifffile.imread('/home/bruno/Documentos/Proyectos/hsimel/experimentos/melanomas/sp/' + names[i])
        phasor = phlib.phasor_tile(im, 1024, 1024)  # Calculo el phasor del tile
        # Concateno las 5 imagenes que calcula phasor_tile
        aux = []
        for j in range(3):
            aux.append(phlib.concatenate(phasor[j], int(names[i][10:12]), int(names[i][13:15]),
                                         bidirectional=False, hper=0.07))
        aux = np.asarray(aux)
        # Filtro con la mediana G y S para limpiar el phasor
        for k in range(3):
            aux[1] = median(aux[1])
            aux[2] = median(aux[2])
        aux[1] = np.where(aux[0] > 2, aux[1], np.zeros(aux[1].shape))
        aux[2] = np.where(aux[0] > 2, aux[2], np.zeros(aux[2].shape))

        histmd = np.histogram(np.concatenate(aux[1]), bins=binsmd)[0][1:]
        histph = np.histogram(np.concatenate(aux[2]), bins=binsph)[0][1:]

        imd[i] = phlib.center_of_mass(binsmd[:99], histmd)
        iph[i] = phlib.center_of_mass(binsph[:133], histph)

    df = pd.DataFrame(list(zip(*[imd, iph]))).add_prefix('Col')
    df.to_csv('/home/bruno/Documentos/Proyectos/hsimel/experimentos/melanomas/melanomas.csv', index=False)


    datam = pd.read_csv("/home/bruno/Documentos/Proyectos/hsimel/experimentos/melanomas/melanomas.csv")
    datan = pd.read_csv("/home/bruno/Documentos/Proyectos/hsimel/experimentos/nevos/nevo.csv")


    plt.figure(1)
    plt.plot(datam["Col0"], datam["Col1"], 'ro', label='melanoma')
    plt.plot(datan["Col0"], datan["Col1"], 'bo', label='nevo')
    plt.legend()
    plt.grid()
    plt.show()
