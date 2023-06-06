import numpy as np
import tifffile
import matplotlib.pyplot as plt
import phasorlibrary as phlib
import os
import pandas as pd
from phasorlibrary import confidence_ellipse
from scipy.stats import ttest_ind

cal = False
if cal:
    binsph = np.arange(45, 180)
    binsmd = np.linspace(0, 1, 101)
    # Grafico los histogramas de todos los nevos y los melanomas
    fig2, (ax2, ax3) = plt.subplots(1, 2, figsize=(12, 4))

    names = os.listdir('/home/bruno/Documentos/Proyectos/hsimel/Paper/figures/fig4/phasors/mel/')
    imd = np.zeros(len(names))
    iph = np.zeros(len(names))

    for i in range(len(names)):
        im = tifffile.imread('/home/bruno/Documentos/Proyectos/hsimel/Paper/figures/fig4/phasors/mel/' + names[i])

        histmd = np.histogram(im[4].flatten(), bins=binsmd)[0][1:]
        histph = np.histogram(im[3].flatten(), bins=binsph)[0][1:]

        imd[i] = phlib.center_of_mass(binsmd[:99], histmd)
        iph[i] = phlib.center_of_mass(binsph[:133], histph)

        # df = pd.DataFrame(list(zip(*[imd, iph]))).add_prefix('Col')
        # df.to_csv('/home/bruno/Documentos/Proyectos/hsimel/Paper/figures/fig4/cmm.csv', index=False)

        ax2.plot(binsmd[:99], histmd, 'r')
        ax2.set_yscale('log')
        ax2.set_xlabel('Modulation')
        ax3.plot(binsph[:133], histph, 'r')
        ax3.set_yscale('log')
        ax3.set_xlabel('Phase')

    names = os.listdir('/home/bruno/Documentos/Proyectos/hsimel/Paper/figures/fig4/phasors/nevo/')
    imd = np.zeros(len(names))
    iph = np.zeros(len(names))
    for i in range(len(names)):
        im = tifffile.imread('/home/bruno/Documentos/Proyectos/hsimel/Paper/figures/fig4/phasors/nevo/' + names[i])

        histmd = np.histogram(im[4].flatten(), bins=binsmd)[0][1:]
        histph = np.histogram(im[3].flatten(), bins=binsph)[0][1:]

        imd[i] = phlib.center_of_mass(binsmd[:99], histmd)
        iph[i] = phlib.center_of_mass(binsph[:133], histph)

        # df = pd.DataFrame(list(zip(*[imd, iph]))).add_prefix('Col')
        # df.to_csv('/home/bruno/Documentos/Proyectos/hsimel/Paper/figures/fig4/cmm.csv', index=False)

        ax2.plot(binsmd[:99], histmd, 'b')
        ax2.set_yscale('log')
        ax2.set_xlabel('Modulation')
        ax3.plot(binsph[:133], histph, 'b')
        ax3.set_yscale('log')
        ax3.set_xlabel('Phase [Degree]')
    ax2.grid()
    ax3.grid()

# Grafico la Elipse de confidencia con los centros de masa
plotty = False
if plotty:
    datam = pd.read_csv('/home/bruno/Documentos/Proyectos/hsimel/Paper/figures/fig4/cmm.csv')
    datan = pd.read_csv('/home/bruno/Documentos/Proyectos/hsimel/Paper/figures/fig4/cmn.csv')

    fig, ax = plt.subplots(1)
    plt.plot(datan["Col0"], datan["Col1"], 'b*', label='Nevus')
    plt.plot(datam["Col0"], datam["Col1"], 'ro', label='Melanoma')
    plt.xlim([0.2, 0.8])
    plt.ylim([60, 140])
    plt.legend()
    plt.xlabel('Modulation')
    plt.ylabel('Phase')
    plt.title('Confidence Ellipse')
    plt.grid()
    confidence_ellipse(datam["Col0"], datam["Col1"], ax, n_std=3, edgecolor='red')
    confidence_ellipse(datam["Col0"], datam["Col1"], ax, n_std=2, edgecolor='red')
    confidence_ellipse(datam["Col0"], datam["Col1"], ax, n_std=1, edgecolor='red')

    confidence_ellipse(datan["Col0"], datan["Col1"], ax, n_std=3, edgecolor='blue')
    confidence_ellipse(datan["Col0"], datan["Col1"], ax, n_std=2, edgecolor='blue')
    confidence_ellipse(datan["Col0"], datan["Col1"], ax, n_std=1, edgecolor='blue')

    plt.figure(4)
    plt.plot(np.ones(len(datan["Col0"])), datan["Col0"], 'b*', label='Nevus')
    plt.plot(2 * np.ones(len(datam["Col0"])), datam["Col0"], 'ro', label='Melanoma')
    plt.legend()
    plt.ylabel('Modulation')
    plt.grid()

    plt.figure(5)
    plt.plot(np.ones(len(datan["Col1"])), datan["Col1"], 'b*', label='Nevus')
    plt.plot(2 * np.ones(len(datam["Col1"])), datam["Col1"], 'ro', label='Melanoma')
    plt.legend()
    plt.ylabel('Phase')
    plt.grid()

    plt.show()

ttest = True
if ttest:
    datam = pd.read_csv('/home/bruno/Documentos/Proyectos/hsimel/Paper/figures/fig4/cmm.csv')
    datan = pd.read_csv('/home/bruno/Documentos/Proyectos/hsimel/Paper/figures/fig4/cmn.csv')
    tm, pm = ttest_ind(datam["Col0"], datan["Col0"], equal_var=False)
    tp, pp = ttest_ind(datam["Col1"], datan["Col1"], equal_var=False)
    print('p mod =', pm, 'p phase=', pp)
