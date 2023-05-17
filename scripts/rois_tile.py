import numpy as np
from hsipy import hsitools, hsi_visualization
import tifffile
import matplotlib.pyplot as plt
from skimage.filters import median
import phasorlibrary as phlib
from matplotlib import colors

# im = tifffile.imread('/home/bruno/Documentos/Proyectos/hsimel/experimentos/melanomas/sp/sp_15116_r1.lsm')
# dc, g, s, md, ph = hsitools.phasor(im)

im = tifffile.imread('/home/bruno/Documentos/Proyectos/hsimel/experimentos/nevos/sp/'
                     'nevo_15361_04x02.lsm')
dc, g, s, md, ph = hsitools.tilephasor(im, dimx=1024, dimy=1024)

x1 = 1400
x2 = 1850
y1 = 630
y2 = 1800

concat = True
if concat:
    n = 2
    m = 4
    b = False
    dc = hsitools.tile_stitching(dc, n, m, bidirectional=b)
    g = hsitools.tile_stitching(g, n, m, bidirectional=b)
    s = hsitools.tile_stitching(s, n, m, bidirectional=b)
    ph = hsitools.tile_stitching(ph, n, m, bidirectional=b)
    md = hsitools.tile_stitching(md, n, m, bidirectional=b)

roi = False
if roi:
    dc2 = dc[x1:x2, y1:y2]
    g2 = g[x1:x2, y1:y2]
    s2 = s[x1:x2, y1:y2]
    ph2 = ph[x1:x2, y1:y2]
    md2 = md[x1:x2, y1:y2]

    hsi_visualization.interactive1(dc2, g2, s2, 0.1, nbit=8,
                                   ncomp=5, histeq=True, filt=True, nfilt=3,
                                   spectrums=False, hsi_stack=np.asarray(im),
                                   lamd=np.arange(415, 715, 10))

ver = False
if ver:  # el codigo de abajo permite usar la funcion interactiva 1
    hsi_visualization.interactive1(dc, g, s, 0.1, nbit=8,
                                   ncomp=5, histeq=True, filt=True, nfilt=3,
                                   spectrums=False, hsi_stack=np.asarray(im),
                                   lamd=np.arange(415, 715, 10))

# defino las regiones que voy a usar, grafico el phasor si quiero y las guardo como ome.tiff
store = False
if store:
    dc2 = dc[x1:x2, y1:y2]
    g2 = g[x1:x2, y1:y2]
    s2 = s[x1:x2, y1:y2]
    ph2 = ph[x1:x2, y1:y2]
    md2 = md[x1:x2, y1:y2]

    ic = 3
    filtnum = 3

    for i in range(filtnum):
        dc2 = median(dc2)
        g2 = median(g2)
        s2 = median(s2)
        ph2 = median(ph2)
        md2 = median(md2)

    dc2 = np.where(dc2 > ic, dc2, np.nan * dc2)
    g2 = np.where(dc2 > ic, g2, np.nan * g2)
    s2 = np.where(dc2 > ic, s2, np.nan * s2)
    ph2 = np.where(dc2 > ic, ph2, np.nan * ph2)
    md2 = np.where(dc2 > ic, md2, np.nan * md2)

    save = True
    if save:
        phlib.generate_file('/home/bruno/Documentos/Proyectos/hsimel/experimentos/rois/roi_nevo_15361',
                            np.asarray([dc2, g2, s2, ph2, md2]))

    plotty = True
    if plotty:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.hist2d(g2.flatten(), s2.flatten(), bins=256, cmap="RdYlGn_r",
                  norm=colors.LogNorm(), range=[[-1, 1], [-1, 1]])
        phlib.phasor_circle(ax)
        ax.set_title('Nevo 15361')
        plt.show()
