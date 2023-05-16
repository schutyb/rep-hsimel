import numpy as np
from hsipy import hsitools, hsi_visualization
import tifffile
import os
import matplotlib.pyplot as plt

# im = tifffile.imread('/home/bruno/Documentos/Proyectos/hsimel/datos/rois/regiones/mel/23061/SP_23061_r2.lsm')
im = tifffile.imread('/home/bruno/Documentos/Proyectos/hsimel/experimentos/nevos/sp/nevo_15397_02x06.lsm')
# im = tifffile.imread('/home/bruno/Documentos/Proyectos/hsimel/experimentos/melanomas/sp/sp_15116_r1.lsm')
dc, g, s, md, ph = hsitools.tilephasor(im, dimx=1024, dimy=1024)
# dc, g, s, md, ph = hsitools.phasor(im)
n = 6
m = 2
b = False

dc = hsitools.tile_stitching(dc, n, m, bidirectional=b)
g = hsitools.tile_stitching(g, n, m, bidirectional=b)
s = hsitools.tile_stitching(s, n, m, bidirectional=b)
# ph = hsitools.tile_stitching(ph, n, m, bidirectional=b)
# md = hsitools.tile_stitching(md, n, m, bidirectional=b)

# phint = np.array([90, 160])
# mdint = np.array([0.4, 0.8])
# hsi_visualization.interactive2(dc, g, s, 8, ph, phint, modulation=md, mdint=mdint,
# histeq=False, filt=True, nfilt=3)

hsi_visualization.interactive1(dc, g, s, 0.1, nbit=8,
                               ncomp=5, histeq=True, filt=True, nfilt=3,
                               spectrums=False, hsi_stack=np.asarray(im),
                               lamd=np.arange(415, 715, 10))
