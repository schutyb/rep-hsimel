import numpy as np
import tifffile
import matplotlib.pyplot as plt
from skimage.exposure import equalize_adapthist
from matplotlib import colors
import phasorlibrary as phlib


# Calculo los histogramas de md y ph
imm = tifffile.imread('/home/bruno/Documentos/Proyectos/hsimel/Paper/figures/fig2y3/'
                      'mel_18852_06x06.ome.tiff')
x1, x2, y1, y2 = 1600, 4400, 430, 1030
dcm, gm, sm, mdm, phm = imm[0][x1:x2, y1:y2], imm[1][x1:x2, y1:y2], imm[2][x1:x2, y1:y2], \
                        imm[3][x1:x2, y1:y2], imm[4][x1:x2, y1:y2]

imn = tifffile.imread('/home/bruno/Documentos/Proyectos/hsimel/Paper/figures/fig2y3/'
                      'nevo_16256_04x02.ome.tiff')
x1, x2, y1, y2 = 1050, 1400, 200, 2800
dcn, gn, sn, mdn, phn = imn[0][x1:x2, y1:y2], imn[1][x1:x2, y1:y2], imn[2][x1:x2, y1:y2], \
                        imn[3][x1:x2, y1:y2], imn[4][x1:x2, y1:y2]

binsph = np.arange(45, 180)
binsmd = np.linspace(0, 1, 101)

histmdm = np.histogram(mdm.flatten(), bins=binsmd)[0][1:]
histphm = np.histogram(phm.flatten(), bins=binsph)[0][1:]

histmdn = np.histogram(mdn.flatten(), bins=binsmd)[0][1:]
histphn = np.histogram(phn.flatten(), bins=binsph)[0][1:]

plt.figure(1)
plt.imshow(equalize_adapthist(dcm / dcm.max()), cmap='gray')
plt.axis('off')

plt.figure(2)
plt.imshow(equalize_adapthist(dcn / dcn.max()), cmap='gray')
plt.axis('off')

# Grafico los phasors
fig1, ax1 = plt.subplots(figsize=(8, 8))
ax1.hist2d(gm.flatten(), sm.flatten(), bins=256, cmap="RdYlGn_r",
           norm=colors.LogNorm(), range=[[-1, 1], [-1, 1]])
phlib.add_color_wheel(ax1)
phlib.phasor_circle(ax1)

fig4, ax4 = plt.subplots(figsize=(8, 8))
ax4.hist2d(gn.flatten(), sn.flatten(), bins=256, cmap="RdYlGn_r",
           norm=colors.LogNorm(), range=[[-1, 1], [-1, 1]])
phlib.add_color_wheel(ax4)
phlib.phasor_circle(ax4)

fig2, (ax2, ax3) = plt.subplots(1, 2, figsize=(12, 4))
ax2.plot(binsmd[:99], histmdm, 'r', binsmd[:99], histmdn, 'b')
ax2.set_yscale('log')
ax2.legend(['Melanoma', 'Nevo'])
ax2.set_xlabel('Modulation')
ax2.grid()

ax3.plot(binsph[:133], histphm, 'r', binsph[:133], histphn, 'b')
ax3.set_yscale('log')
ax3.legend(['Melanoma', 'Nevo'])
ax3.set_xlabel('Phase')
ax3.grid()
plt.show()
