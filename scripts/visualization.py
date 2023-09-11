""" Script for visualization pipeline"""

import numpy as np
from hsipy import hsitools, hsi_visualization
import tifffile


im = tifffile.imread('/home/bruno/Documentos/Proyectos/hsimel/Paper/data/spectral/lsm/mel_18852_06x06.lsm')
dc, g, s, md, ph = np.asarray(hsitools.tilephasor(im, 1024, 1024))
scan_dir = False
dx = 6
dy = 6
dc = hsitools.tile_stitching(dc, dx, dy, bidirectional=scan_dir)
g = hsitools.tile_stitching(g, dx, dy, bidirectional=scan_dir)
s = hsitools.tile_stitching(s, dx, dy, bidirectional=scan_dir)
md = hsitools.tile_stitching(md, dx, dy, bidirectional=scan_dir)
ph = hsitools.tile_stitching(ph, dx, dy, bidirectional=scan_dir)


interactive = True
if interactive:
    hsi_visualization.interactive1(dc, g, s, 0.25, 8, ncomp=3, nfilt=3, spectrums=False,
                                   hsi_stack=im, lamd=np.linspace(418, 718, 29))
