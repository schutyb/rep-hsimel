import numpy as np
import tifffile
import hsipy
import matplotlib.pyplot as plt
from skimage.filters import median
from skimage.exposure import equalize_adapthist
from matplotlib import colors
import phasorlibrary as phlib


im = tifffile.imread('/home/bruno/Documentos/Proyectos/hsimel/Paper/data/spectral/lsm/mel_18852_06x06.lsm')
dc, g, s, md, ph = np.asarray(hsipy.hsitools.tilephasor(im, dimx=1024, dimy=1024))
dx = 6
dy = 6

dc = hsipy.hsitools.tile_stitching(dc, dx, dy, bidirectional=False)
g = hsipy.hsitools.tile_stitching(g, dx, dy, bidirectional=False)
s = hsipy.hsitools.tile_stitching(s, dx, dy, bidirectional=False)
md = hsipy.hsitools.tile_stitching(md, dx, dy, bidirectional=False)
ph = hsipy.hsitools.tile_stitching(ph, dx, dy, bidirectional=False)

# Filtro con la mediana G y S para limpiar el phasor
for k in range(3):
    dc = median(dc)
    g = median(g)
    s = median(s)
    md = median(md)
    ph = median(ph)

# umbralizo para sacar el fondo
g = np.where(dc > 5, g, dc * np.nan)
s = np.where(dc > 5, s, dc * np.nan)
ph = np.where(dc > 5, ph, dc * np.nan)
md = np.where(dc > 5, md, dc * np.nan)

g = np.where(ph > 45, g, dc * np.nan)
g = np.where(ph < 180, g, dc * np.nan)

s = np.where(ph > 45, s, dc * np.nan)
s = np.where(ph < 180, s, dc * np.nan)

g = np.where(md > 0.5, g, dc * np.nan)
g = np.where(md < 0.7, g, dc * np.nan)

s = np.where(md > 0.5, s, dc * np.nan)
s = np.where(md < 0.7, s, dc * np.nan)

# md = np.where(dc > 5, md, dc * np.nan)
# ph = np.where(dc > 5, ph, dc * np.nan)

# rgb = phlib.colored_image(ph, np.asarray([45, 180]), outlier_cut=True, color_scale=0.95)

plt.figure(1)
plt.imshow(equalize_adapthist(dc / dc.max()), cmap='gray')
plt.axis('off')
plt.title('Melanoma')

# Grafico los phasors
fig1, ax1 = plt.subplots(figsize=(8, 8))
ax1.hist2d(g.flatten(), s.flatten(), bins=256, cmap="RdYlGn_r",
           norm=colors.LogNorm(), range=[[-1, 1], [-1, 1]])
phlib.add_color_wheel(ax1)
phlib.phasor_circle(ax1)

# plt.figure(3)
# plt.imshow(rgb)
# plt.axis('off')
plt.show()
