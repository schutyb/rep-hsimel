import numpy as np
import tifffile
import hsipy
import matplotlib.pyplot as plt
from skimage.filters import median
from skimage.exposure import equalize_adapthist
from matplotlib import colors
import phasorlibrary as phlib
import pandas as pd


path = '/home/bruno/Documentos/Proyectos/hsimel/Paper/figures/fig6/'
imn = tifffile.imread(path + '16256-SP-ROI-02-4avg-63x-gain600.lsm')
imm = tifffile.imread(path + '18852_SP_63x_ROI_1_gain600_superficial.lsm')

dcn, gn, sn, mdn, phn = hsipy.hsitools.phasor(imn)
dcm, gm, sm, mdm, phm = hsipy.hsitools.phasor(imm)

# filtreo y umbralizo
for k in range(5):
    dcn = median(dcn)
    gn = median(gn)
    sn = median(sn)
    mdn = median(mdn)
    phn = median(phn)

    dcm = median(dcm)
    gm = median(gm)
    sm = median(sm)
    mdm = median(mdm)
    phm = median(phm)

gn = np.where(dcn > 5, gn, dcn * np.nan)
sn = np.where(dcn > 5, sn, dcn * np.nan)
mdn = np.where(dcn > 5, mdn, dcn * np.nan)
phn = np.where(dcn > 5, phn, dcn * np.nan)

gm = np.where(dcm > 5, gm, dcn * np.nan)
sm = np.where(dcm > 5, sm, dcn * np.nan)
mdm = np.where(dcm > 5, mdm, dcn * np.nan)
phm = np.where(dcm > 5, phm, dcn * np.nan)

# hsipy.hsi_visualization.interactive1(dcn, gn, sn, 0.1, nbit=8, ncomp=5, histeq=True, filt=True, nfilt=5)

# calculo la imagen de pseudocolor
rgbn = phlib.colored_image(phn, np.asarray([45, 180]), outlier_cut=True, color_scale=0.95)
rgbm = phlib.colored_image(phm, np.asarray([45, 180]), outlier_cut=True, color_scale=0.95)

# calculo los histogramas
binsph = np.arange(45, 200)
binsmd = np.linspace(0, 1, 101)
histmdn = np.histogram(mdn.flatten(), bins=binsmd)[0][1:]
histphn = np.histogram(phn.flatten(), bins=binsph)[0][1:]
histmdm = np.histogram(mdm.flatten(), bins=binsmd)[0][1:]
histphm = np.histogram(phm.flatten(), bins=binsph)[0][1:]

plotty = True
if plotty:
    cols = ['r', 'orange', 'yellow', 'chartreuse', 'olivedrab', 'cyan', 'b', 'purple']
    names = ['Colageno_II', 'Colageno_IV', 'Colageno_III', 'Colageno_I', 'NADH',
             'Elastin', 'Melanin', 'FAD+']
    cmcomp = pd.read_csv('/home/bruno/Documentos/Proyectos/hsimel/Paper/figures/fig5/cm_components.csv')

    plt.figure(1)
    plt.imshow(equalize_adapthist(dcn / dcn.max()), cmap='gray')
    plt.axis('off')
    plt.title('Nevo')

    plt.figure(2)
    plt.imshow(equalize_adapthist(dcm / dcm.max()), cmap='gray')
    plt.axis('off')
    plt.title('Melanoma')

    # Grafico los phasors
    fig1, ax1 = plt.subplots(figsize=(8, 8))
    ax1.hist2d(gn.flatten(), sn.flatten(), bins=256, cmap="RdYlGn_r",
               norm=colors.LogNorm(), range=[[-1, 1], [-1, 1]])
    phlib.add_color_wheel(ax1)
    phlib.phasor_circle(ax1)

    fig2, ax2 = plt.subplots(figsize=(8, 8))
    ax2.hist2d(gm.flatten(), sm.flatten(), bins=256, cmap="RdYlGn_r",
               norm=colors.LogNorm(), range=[[-1, 1], [-1, 1]])
    phlib.add_color_wheel(ax2)
    phlib.phasor_circle(ax2)

    for i in range(len(names)):
        circle1 = plt.Circle((cmcomp["Col0"][i], cmcomp["Col1"][i]), 0.02, color=cols[i], label=names[i])
        circle2 = plt.Circle((cmcomp["Col0"][i], cmcomp["Col1"][i]), 0.02, color=cols[i], label=names[i])
        ax1.add_patch(circle1)
        ax2.add_patch(circle2)
        # ax1.legend()
        # ax2.legend()

    # Ploteo la imagen de pseudocolor
    plt.figure(5)
    plt.imshow(rgbn)
    plt.axis('off')

    plt.figure(6)
    plt.imshow(rgbm)
    plt.axis('off')

    plt.figure(7)
    plt.plot(binsmd[:99], histmdn, 'b', label='Nevo')
    plt.yscale('log')
    plt.xlabel('Modulation')
    plt.legend()
    plt.grid()

    plt.figure(8)
    plt.plot(binsmd[:99], histmdm, 'r', label='Melanoma')
    plt.yscale('log')
    plt.xlabel('Modulation')
    plt.legend()
    plt.grid()

    plt.figure(9)
    plt.plot(binsph[:153], histphn, 'b', label='Nevo')
    plt.yscale('log')
    plt.xlabel('Phase')
    plt.legend()
    plt.grid()

    plt.figure(10)
    plt.plot(binsph[:153], histphm, 'r', label='Melanoma')
    plt.yscale('log')
    plt.xlabel('Phase')
    plt.legend()
    plt.grid()
    plt.show()

plot2 = False
if plot2:
    fig2, (ax2, ax3) = plt.subplots(1, 2, figsize=(12, 4))
    ax2.plot(binsmd[:99], histmdn, 'b', binsmd[:99], histmdm, 'r')
    ax2.set_yscale('log')
    ax2.set_xlabel('Modulation')
    ax2.legend(['Nevo', 'Melanoma'])
    ax2.grid()

    ax3.plot(binsph[:153], histphn, 'b', binsph[:153], histphm, 'r')
    ax3.set_yscale('log')
    ax3.set_xlabel('Phase [Degrees]')
    ax3.legend(['Nevo', 'Melanoma'])
    ax3.grid()
    plt.show()

