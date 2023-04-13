import numpy as np
import tifffile
import phasorlibrary as phlib
import matplotlib.pyplot as plt
from matplotlib import colors
from skimage.exposure import equalize_adapthist
import pandas as pd


'''FIGURA 5: DC, PHASOR CON COMPONENTES, PSEUDOCOLOR E HISTOGRAMAS DE FASE Y MODULACION'''
fig5plot = True
if fig5plot:
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
    binsmd = np.linspace(0, 1, 100)

    histmdn = np.histogram(np.concatenate(imn[3]), bins=binsmd)[0][1:]
    histphn = np.histogram(np.concatenate(imn[4]), bins=binsph)[0][1:]

    histmdnp = np.histogram(np.concatenate(imnp[3]), bins=binsmd)[0][1:]
    histphnp = np.histogram(np.concatenate(imnp[4]), bins=binsph)[0][1:]

    histmdm = np.histogram(np.concatenate(imm[3]), bins=binsmd)[0][1:]
    histphm = np.histogram(np.concatenate(imm[4]), bins=binsph)[0][1:]

    # leo las coordenadas de las componentes
    coord = np.asarray(pd.read_csv('/home/bruno/Documentos/Proyectos/hsimel/datos/componentes/cm_melanocitos.csv'))
    c2 = ['red', 'purple', 'blue', 'black', 'orange']
    ic = 2  # threshold intensity

    fig, ((ax1, ax2, ax3, axh1, axhm1), (ax4, ax5, ax6, axh2, axhm2),
          (ax7, ax8, ax9, axh3, axhm3)) = plt.subplots(3, 5, figsize=(20, 10))

    # Nevo
    ax1.imshow(equalize_adapthist(dcn / dcn.max()), cmap='gray')
    ax1.axis('off')
    xn, yn = phlib.histt(imn[0], imn[1], imn[2], ic)
    ax2.hist2d(xn, yn, bins=256, cmap="RdYlGn_r", norm=colors.LogNorm(),
               range=[[-1, 1], [-1, 1]])
    ax2.set_xlabel('G')
    ax2.set_ylabel('S')
    phlib.phasor_circle(ax2)
    ax3.axis('off')
    ax3.imshow(rgbn)
    axh1.plot(binsph[:133], histphn / histphn.max(), 'r')
    axh1.set_yscale('log')
    axhm1.plot(binsmd[:98], histmdn / histmdn.max())
    axhm1.set_yscale('log')
    axhm1.grid()
    axh1.grid()

    # Nevo pig
    ax4.imshow(equalize_adapthist(dcnp / dcnp.max()), cmap='gray')
    ax4.axis('off')
    xnp, ynp = phlib.histt(imnp[0], imnp[1], imnp[2], ic)
    ax5.hist2d(xnp, ynp, bins=256, cmap="RdYlGn_r", norm=colors.LogNorm(),
               range=[[-1, 1], [-1, 1]])
    ax5.set_xlabel('G')
    ax5.set_ylabel('S')
    phlib.phasor_circle(ax5)
    ax6.axis('off')
    ax6.imshow(rgbnp)
    axh2.plot(binsph[:133], histphnp / histphnp.max(), 'r')
    axh2.set_yscale('log')
    axhm2.plot(binsmd[:98], histmdnp / histmdnp.max())
    axhm2.set_yscale('log')
    axhm2.grid()
    axh2.grid()

    # Melanoma
    ax7.imshow(equalize_adapthist(dcm / dcm.max()), cmap='gray')
    ax7.axis('off')
    xm, ym = phlib.histt(imm[0], imm[1], imm[2], ic)
    ax8.hist2d(xm, ym, bins=256, cmap="RdYlGn_r", norm=colors.LogNorm(),
               range=[[-1, 1], [-1, 1]])
    ax8.set_xlabel('G')
    ax8.set_ylabel('S')
    phlib.phasor_circle(ax8)
    ax9.axis('off')
    ax9.imshow(rgbm)
    axh3.plot(binsph[:133], histphm / histphm.max(), 'r')
    axh3.set_yscale('log')
    axh3.set_xlabel('Degrees')
    axhm3.plot(binsmd[:98], histmdm / histmdm.max())
    axhm3.set_yscale('log')
    axhm3.grid()
    axh3.grid()
    fig.tight_layout()

    for i in range(len(c2)):
        circle1 = plt.Circle((coord[i][0], coord[i][1]), 0.025, color=c2[i])  # agrego las componentes al phasor
        ax2.add_patch(circle1)
        circle2 = plt.Circle((coord[i][0], coord[i][1]), 0.025, color=c2[i])
        ax5.add_patch(circle2)
        circle3 = plt.Circle((coord[i][0], coord[i][1]), 0.025, color=c2[i])
        ax8.add_patch(circle3)
    plt.show()
