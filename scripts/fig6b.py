import numpy as np
from hsipy import hsitools, hsi_visualization
import tifffile
import matplotlib.pyplot as plt
from skimage.filters import median
import phasorlibrary as phlib
from matplotlib import colors
import os
from sklearn.cluster import KMeans
import pandas as pd
import confidence_ellipse as confe


cal = False
if cal:
    names = os.listdir('/home/bruno/Documentos/Proyectos/hsimel/phasors/nevo/')
    imd = np.zeros(len(names))
    iph = np.zeros(len(names))

    for i in range(len(names)):
        im = tifffile.imread('/home/bruno/Documentos/Proyectos/hsimel/phasors/nevo/' + names[i])

        binsph = np.arange(45, 180)
        binsmd = np.linspace(0, 1, 101)
        histmd = np.histogram(im[4].flatten(), bins=binsmd)[0][1:]
        histph = np.histogram(im[3].flatten(), bins=binsph)[0][1:]

        imd[i] = phlib.center_of_mass(binsmd[:99], histmd)
        iph[i] = phlib.center_of_mass(binsph[:133], histph)

        df = pd.DataFrame(list(zip(*[imd, iph]))).add_prefix('Col')
        df.to_csv('/home/bruno/Documentos/Proyectos/hsimel/cmn.csv', index=False)

plotty = True
if plotty:
    datam = pd.read_csv('/home/bruno/Documentos/Proyectos/hsimel/cmm.csv')
    datan = pd.read_csv('/home/bruno/Documentos/Proyectos/hsimel/cmn.csv')

    fig, ax = plt.subplots(1)
    plt.plot(datam["Col0"], datam["Col1"], 'ro', label='melanoma')
    plt.plot(datan["Col0"], datan["Col1"], 'bo', label='nevo')
    plt.xlim([0.2, 0.8])
    plt.ylim([60, 140])
    plt.legend()
    plt.grid()
    confe.confidence_ellipse(datam["Col0"], datam["Col1"], ax, edgecolor='red')
    confe.confidence_ellipse(datan["Col0"], datan["Col1"], ax, edgecolor='blue')

plt.show()
