import numpy as np
from hsipy import hsitools, hsi_visualization
import tifffile
import os

im = None
dc = None
g = None
s = None
notile = True
if notile:
    im = tifffile.imread('/home/bruno/Documentos/Proyectos/hsimel/datos/mel/sp_12635_r1.lsm')
    dc, g, s, md, ph = hsitools.phasor(im, harmonic=1)

tile = False
if tile:
    im = tifffile.imread('/home/bruno/Documentos/Proyectos/hsimel/experimentos/melanomas/sp/mel_12635_12x05.lsm')
    dc, g, s, md, ph = hsitools.tilephasor(im, dimx=1024, dimy=1024)
    n = 5
    m = 12
    b = False

    dc = hsitools.tile_stitching(dc, n, m, bidirectional=b)
    g = hsitools.tile_stitching(g, n, m, bidirectional=b)
    s = hsitools.tile_stitching(s, n, m, bidirectional=b)
    md = hsitools.tile_stitching(md, n, m, bidirectional=b)
    ph = hsitools.tile_stitching(ph, n, m, bidirectional=b)

hsi_visualization.interactive1(dc, g, s, 0.15, nbit=8, ncomp=3, histeq=True, filt=True, nfilt=5,
                               spectrums=True, hsi_stack=np.asarray(im), lamd=np.arange(415, 715, 10))

testall = False
if testall:
    names = os.listdir('/home/bruno/Documentos/Proyectos/hsimel/datos/rois/regiones/mel/phasors/')
    for k in range(20):
        dc, g, s, _, _ = tifffile.imread('/home/bruno/Documentos/Proyectos/hsimel/datos/rois/regiones/mel/phasors/' +
                                         names[k])
        hsi_visualization.interactive1(dc, g, s, 0.1, nbit=8, ncomp=3, histeq=False)

    # hsi_visualization.interactive2(dc, g, s, phase=ph, phint=np.asarray([45, 180]), modulation=md,
    # mdint=np.asarray([0, 0.6]), nbit=8, nfilt=3, filt=True)

