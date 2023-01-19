"""This script perform the phasor of the full image tile,
for the 3 cases we are using"""

import numpy as np
import tifffile
import phasorlibrary as ph
import matplotlib.pyplot as plt
from skimage.filters import median

# Leo los archivos
imnevo = tifffile.imread('/home/bruno/Documentos/Proyectos/hsimel/datos/SP_16256_6x3_bidir_gain600_4avg.lsm')
# imnevo = tifffile.imread('/home/bruno/Documentos/Proyectos/hsimel/datos/SP_18852_10x10_bidir_autofocus.lsm')

# Calculo el phasor del tile
phasornevo = ph.phasor_tile(imnevo, 1024, 1024)

# Concateno las 5 imagenes que calcula phasor_tile
nevo = []
for i in range(5):
    nevo.append(ph.concatenate(phasornevo[i], 3, 6, bidirectional=True))
nevo = np.asarray(nevo)

# Filtro con la mediana G y S para limpiar el phasor
for i in range(3):
    nevo[1] = median(nevo[1])
    nevo[2] = median(nevo[2])

plotty = False
if plotty:
    plt.figure(1)
    plt.imshow(1, cmap='gray')
    plt.show()
