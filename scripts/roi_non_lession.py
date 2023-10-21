import numpy as np
import tifffile
import matplotlib.pyplot as plt
from skimage.exposure import equalize_adapthist
from matplotlib import colors
import phasorlibrary as phlib
from hsipy import hsitools
from skimage.filters import median

# Calculo los histogramas de md y ph
im = tifffile.imread('/home/bruno/Documentos/Proyectos/hsimel/datos/'
                     'nevo_15361_04x02.lsm')
dc, g, s, md, ph = np.asarray(hsitools.tilephasor(im, dimx=1024, dimy=1024))
dx = 2
dy = 4
scan_dir = False

dc = hsitools.tile_stitching(dc, dx, dy, bidirectional=scan_dir)
g = hsitools.tile_stitching(g, dx, dy, bidirectional=scan_dir)
s = hsitools.tile_stitching(s, dx, dy, bidirectional=scan_dir)
md = hsitools.tile_stitching(md, dx, dy, bidirectional=scan_dir)
ph = hsitools.tile_stitching(ph, dx, dy, bidirectional=scan_dir)

# Filtro con la mediana G y S para limpiar el phasor
for k in range(3):
    g = median(g)
    s = median(s)

mask1 = np.ones(dc.shape)
mask0 = np.zeros(dc.shape) * np.nan
mask = np.where(ph > 75, mask1, mask0) * np.where(ph < 180, mask1, mask0) * \
       np.where(md > 0.3, mask1, mask0) * np.where(md < 0.9, mask1, mask0) * \
       np.where(dc > 5, mask1, mask0)

g = g * mask
s = s * mask
aux = s.flatten() * g.flatten()
gvec = g.flatten()
svec = s.flatten()
gvec = gvec[~np.isnan(aux)]
svec = svec[~np.isnan(aux)]

md = md * mask
ph = ph * mask
aux = md.flatten() * ph.flatten()
mdvec = md.flatten()
phvec = ph.flatten()
mdvec = mdvec[~np.isnan(aux)]
phvec = phvec[~np.isnan(aux)]

x1, x2, y1, y2 = 1630, 1860, 370, 590
dcm, gm, sm, mdm, phm = dc[x1:x2, y1:y2], g[x1:x2, y1:y2], s[x1:x2, y1:y2], \
                        md[x1:x2, y1:y2], ph[x1:x2, y1:y2]

binsph = np.arange(45, 180)
binsmd = np.linspace(0, 1, 101)

histmdm = np.histogram(mdm.flatten(), bins=binsmd)[0][1:]
histphm = np.histogram(phm.flatten(), bins=binsph)[0][1:]

dcaux = equalize_adapthist(dc / dc.max())

plt.figure(1)
plt.imshow(dcaux, cmap='gray')
plt.plot(y1, x1, 'ro')
plt.plot(y2, x2, 'ro')
plt.axis('off')

# Grafico los phasors
fig1, ax1 = plt.subplots(figsize=(8, 8))
ax1.hist2d(gm.flatten(), sm.flatten(), bins=256, cmap="RdYlGn_r",
           norm=colors.LogNorm(), range=[[-1, 1], [-1, 1]])
phlib.add_color_wheel(ax1)
phlib.phasor_circle(ax1)

fig2, (ax2, ax3) = plt.subplots(1, 2, figsize=(12, 4))
ax2.plot(binsmd[:99], histmdm, 'r')
ax2.set_yscale('log')
ax2.set_xlabel('Modulation')
ax2.grid()

ax3.plot(binsph[:133], histphm, 'r')
ax3.set_yscale('log')
ax3.set_xlabel('Phase')
ax3.grid()
plt.show()
