import numpy as np
import tifffile
import phasorlibrary as phlib
import matplotlib.pyplot as plt
from matplotlib import colors
from skimage.filters import median
import os
from skimage.exposure import equalize_adapthist
from matplotlib.pyplot import figure
from sklearn.cluster import KMeans


''' GRAFICO EL HSI EL PHASOR Y LA PSEUDO COLOR'''
rois = False
if rois:
    imm = tifffile.imread('/home/bruno/Documentos/Proyectos/hsimel/datos/rois/rois2/phasors/mel/melp_001.ome.tiff')
    dcm = imm[0]  # imagen de intensidad promedio que representa el HSI
    rgbm = phlib.colored_image(imm[4], np.asarray([45, 180]), outlier_cut=False, color_scale=1)

    imn = tifffile.imread('/home/bruno/Documentos/Proyectos/hsimel/datos/rois/rois2/phasors/nevo/nevo_001.ome.tiff')
    dcn = imn[0]  # imagen de intensidad promedio que representa el HSI
    rgbn = phlib.colored_image(imn[4], np.asarray([45, 180]), outlier_cut=False, color_scale=1)

    imnp = tifffile.imread('/home/bruno/Documentos/Proyectos/hsimel/datos/rois/rois2/phasors/nevop/nevop_01.ome.tiff')
    dcnp = imnp[0]  # imagen de intensidad promedio que representa el HSI
    rgbnp = phlib.colored_image(imnp[4], np.asarray([45, 180]), outlier_cut=False, color_scale=1)

    binsph = np.arange(45, 180)
    # binsmd = np.linspace(0, 1, 100)

    # histmdn = np.histogram(np.concatenate(imn[3]), bins=binsmd)[0][1:]
    histphn = np.histogram(np.concatenate(imn[4]), bins=binsph)[0][1:]

    # histmdnp = np.histogram(np.concatenate(imnp[3]), bins=binsmd)[0][1:]
    histphnp = np.histogram(np.concatenate(imnp[4]), bins=binsph)[0][1:]

    # histmdm = np.histogram(np.concatenate(imm[3]), bins=binsmd)[0][1:]
    histphm = np.histogram(np.concatenate(imm[4]), bins=binsph)[0][1:]

    fig, ((ax1, ax2, ax3, axh1), (ax4, ax5, ax6, axh2), (ax7, ax8, ax9, axh3)) = plt.subplots(3, 4, figsize=(15, 10))

    # Nevo
    ax1.imshow(equalize_adapthist(dcn / dcn.max()), cmap='gray')
    ax1.axis('off')
    ax2.hist2d(np.concatenate(imn[1]), np.concatenate(imn[2]), bins=256, cmap="RdYlGn_r", norm=colors.LogNorm(),
               range=[[-1, 1], [-1, 1]])
    phlib.phasor_circle(ax2)
    ax3.axis('off')
    ax3.imshow(rgbn)
    axh1.plot(binsph[:133], histphn / histphn.max(), label='Phase')
    axh1.legend()
    axh1.set_yscale('log')

    # Nevo pig
    ax4.imshow(equalize_adapthist(dcnp / dcnp.max()), cmap='gray')
    ax4.axis('off')
    ax5.hist2d(np.concatenate(imnp[1]), np.concatenate(imnp[2]), bins=256, cmap="RdYlGn_r", norm=colors.LogNorm(),
               range=[[-1, 1], [-1, 1]])
    phlib.phasor_circle(ax5)
    ax6.axis('off')
    ax6.imshow(rgbnp)
    axh2.plot(binsph[:133], histphnp / histphnp.max(), label='Phase')
    axh2.legend()
    axh2.set_yscale('log')

    # Melanoma
    ax7.imshow(equalize_adapthist(dcm/dcm.max()), cmap='gray')
    ax7.axis('off')
    ax8.hist2d(np.concatenate(imm[1]), np.concatenate(imm[2]), bins=256, cmap="RdYlGn_r", norm=colors.LogNorm(),
               range=[[-1, 1], [-1, 1]])
    phlib.phasor_circle(ax8)
    ax9.axis('off')
    ax9.imshow(rgbm)
    axh3.plot(binsph[:133], histphm / histphm.max(), label='Phase')
    axh3.legend()
    axh3.set_yscale('log')

    plt.show()


''' GRAFICO Todas las componentes en el solo phasor'''
componentes = False
if componentes:
    im = tifffile.imread('/home/bruno/Documentos/Proyectos/hsimel/datos/componentes/phasors/02_FAD_1mg_ml.ome.tiff')

    X1 = np.zeros([2, len(np.concatenate(im[1]))])
    X1[0:, 0:] = np.concatenate(im[1]), np.concatenate(im[2])
    X = X1.T
    cluster = KMeans(n_clusters=1).fit(X)
    coordx, coordy = cluster.cluster_centers_[0][0], cluster.cluster_centers_[0][1]

    circle1 = plt.Circle((coordx, coordy), 0.03, color='r')
    fig, ax = plt.subplots(figsize=(8, 8))
    phlib.phasor_circle(ax)
    ax.add_patch(circle1)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    plt.show()
