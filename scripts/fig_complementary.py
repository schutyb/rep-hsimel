import numpy as np
import tifffile
import hsipy
import matplotlib.pyplot as plt
from skimage.filters import median
from skimage.exposure import equalize_adapthist
from matplotlib import colors
import phasorlibrary as phlib

im = tifffile.imread('/home/bruno/Downloads/SP_muestra_18852_Tile_6x6_pos_2b.lsm')
dc, g, s, md, ph = np.asarray(hsipy.hsitools.tilephasor(im, dimx=1024, dimy=1024))
dx = 6
dy = 6
scan_dir = False

dc = hsipy.hsitools.tile_stitching(dc, dx, dy, bidirectional=scan_dir)
g = hsipy.hsitools.tile_stitching(g, dx, dy, bidirectional=scan_dir)
s = hsipy.hsitools.tile_stitching(s, dx, dy, bidirectional=scan_dir)
md = hsipy.hsitools.tile_stitching(md, dx, dy, bidirectional=scan_dir)
ph = hsipy.hsitools.tile_stitching(ph, dx, dy, bidirectional=scan_dir)

# Filtro con la mediana G y S para limpiar el phasor
for k in range(5):
    g = median(g)
    s = median(s)

plotty = True
if plotty:
    threshold = True
    if threshold:
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
    else:
        gvec = g.flatten()
        svec = s.flatten()
    # Grafico los phasors
    fig1, ax1 = plt.subplots(figsize=(8, 8))
    ax1.hist2d(gvec, svec, bins=256, cmap="RdYlGn_r",
               norm=colors.LogNorm(), range=[[-1, 1], [-1, 1]])
    phlib.add_color_wheel(ax1)
    phlib.phasor_circle(ax1)
    plt.show()
