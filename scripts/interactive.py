import numpy as np
from hsipy import hsitools, hsi_visualization
import tifffile
import os


im = tifffile.imread('/home/bruno/Documentos/Proyectos/hsimel/datos/rois/regiones/mel/spectral/SP_22767_r07.lsm')
dc, g, s, md, ph = hsitools.phasor(im)
hsi_visualization.interactive1(dc, g, s, 0.1, nbit=8, ncomp=3, histeq=False, filt=True, nfilt=3)

testall = True
if testall:
    names = os.listdir('/home/bruno/Documentos/Proyectos/hsimel/datos/rois/regiones/mel/phasors/')
    for k in range(20):
        dc, g, s, _, _ = tifffile.imread('/home/bruno/Documentos/Proyectos/hsimel/datos/rois/regiones/mel/phasors/' +
                                         names[k])
        hsi_visualization.interactive1(dc, g, s, 0.1, nbit=8, ncomp=3, histeq=False)

    # hsi_visualization.interactive2(dc, g, s, phase=ph, phint=np.asarray([45, 180]), modulation=md,
    # mdint=np.asarray([0, 0.6]), nbit=8, nfilt=3, filt=True)

