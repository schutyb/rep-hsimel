import numpy as np
from hsipy import hsitools, hsi_visualization
from scipy import signal
import matplotlib.pyplot as plt
import phasorlibrary


aux = signal.gaussian(100, std=7)
soporte = np.zeros(3 * len(aux))
w1 = np.copy(soporte)
w2 = np.copy(soporte)
w3 = np.copy(soporte)

# Phase
w1[0:len(aux)] = aux
w2[len(aux):2 * len(aux)] = aux
w3[2 * len(aux):3 * len(aux)] = aux

plt.figure(1)
plt.plot(w1, 'b')
plt.plot(w2, 'g')
plt.plot(w3, 'r')

_, g1, s1, _, _ = hsitools.phasor(w1)
_, g2, s2, _, _ = hsitools.phasor(w2)
_, g3, s3, _, _ = hsitools.phasor(w3)
cm = np.asarray([[g1, s1], [g2, s2], [g3, s3]])
c2 = ['b', 'g', 'r']

fig, ax = plt.subplots()
plt.title('Phase variation')
phasorlibrary.phasor_circle(ax)
for i in range(len(cm)):
    circle1 = plt.Circle((cm[i][0], cm[i][1]), 0.03, color=c2[i])
    ax.add_patch(circle1)


# Modulation
w1 = signal.gaussian(100, std=3)
w2 = signal.gaussian(100, std=15)
w3 = signal.gaussian(100, std=25)

plt.figure(3)
plt.plot(w1, 'b')
plt.plot(w2, 'g')
plt.plot(w3, 'r')

_, g1, s1, _, _ = hsitools.phasor(w1)
_, g2, s2, _, _ = hsitools.phasor(w2)
_, g3, s3, _, _ = hsitools.phasor(w3)
cm = np.asarray([[g1, s1], [g2, s2], [g3, s3]])
c2 = ['b', 'g', 'r']

fig, ax = plt.subplots()
plt.title('Phase variation')
phasorlibrary.phasor_circle(ax)
for i in range(len(cm)):
    circle1 = plt.Circle((cm[i][0], cm[i][1]), 0.03, color=c2[i])
    ax.add_patch(circle1)

# Linear combination
aux = signal.gaussian(100, std=10)
soporte = np.zeros(int(1.5 * len(aux)))
w1 = np.copy(soporte)
w2 = np.copy(soporte)

# Phase
w1[0:len(aux)] = aux
w2[int(len(aux)/3):int(1.33 * len(aux))] = aux
w3 = w1 + w2

plt.figure(5)
plt.plot(w1/max(w1), 'b')
plt.plot(w2/max(w2), 'g')
plt.plot(w3/max(w3) + 0.05, 'r')

_, g1, s1, _, _ = hsitools.phasor(w1)
_, g2, s2, _, _ = hsitools.phasor(w2)
_, g3, s3, _, _ = hsitools.phasor(w3)
cm = np.asarray([[g1, s1], [g2, s2], [g3, s3]])
c2 = ['b', 'g', 'r']

fig, ax = plt.subplots()
plt.title('Phase variation')
phasorlibrary.phasor_circle(ax)
for i in range(len(cm)):
    circle1 = plt.Circle((cm[i][0], cm[i][1]), 0.03, color=c2[i])
    ax.add_patch(circle1)
plt.show()
