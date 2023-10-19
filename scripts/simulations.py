import numpy as np
from hsipy import hsitools, hsi_visualization
from scipy import signal
import matplotlib.pyplot as plt
import phasorlibrary as ph


# Calibracion
# DELTAS
delta = False
if delta:
    # Defino deltas desde la posicon 0 hasta la 30
    soporte = np.zeros(30)
    fig, ax = plt.subplots()
    ph.phasor_circle(ax)
    for i in range(0, len(soporte)):
        dt = np.copy(soporte)
        dt[i] = 1
        _, g1, s1, _, _ = hsitools.phasor(dt)
        cm = np.asarray([[g1, s1]])
        for j in range(len(cm)):
            circle1 = plt.Circle((cm[j][0], cm[j][1]), 0.03)
            ax.add_patch(circle1)

# GAUSSIAN
gaussian_bool = True
if gaussian_bool:
    # Defino deltas desde la posicon 0 hasta la 30
    l = 30
    t = np.zeros(l)
    sig_gauss = signal.gaussian(int(l/10), std=0.8)
    sig_gauss = np.concatenate([np.zeros(27), sig_gauss])
    sig_gauss = np.flip(sig_gauss)
    # fig, ax = plt.subplots(figsize=(10, 10))
    # ph.phasor_circle(ax)
    c = ['indigo', 'rebeccapurple', 'mediumpurple', 'violet', 'navy', 'blue', 'royalblue', 'dodgerblue',
         'deepskyblue', 'cyan', 'mediumturquoise', 'seagreen', 'lime', 'green', 'olive', 'greenyellow', 'yellowgreen',
         'yellow', 'goldenrod', 'darkgoldenrod', 'orange', 'darkorange', 'peru', 'sandybrown',
         'chocolate', 'chocolate', 'tomato', 'red', 'red', 'darkred']
    plt.figure(1)
    for i in range(0, l):
        aux_gauss = np.zeros(l)
        borders = False
        if borders:
            if i == 0:
                aux_gauss[0] = sig_gauss[0]
            elif i == 1:
                aux_gauss[0] = sig_gauss[1]
                aux_gauss[1] = sig_gauss[0]
            elif i == l - 2:
                aux_gauss[l - 2:l] = sig_gauss[0:2]
            elif i == l - 1:
                aux_gauss[l - 1:l] = sig_gauss[0:1]
            else:
                if i == 2:
                    aux_gauss[0:3] = sig_gauss[0:3]
                else:
                    aux_gauss[i:i + 3] = sig_gauss[0:3]
        else:
            # aux_gauss[i:i + 3] = sig_gauss[0:3]
            aux_gauss[i] = 1
        plt.plot(aux_gauss, c[i], linewidth=2)
        _, g1, s1, _, _ = ph.phasor(aux_gauss)
        cm = np.asarray([[g1, s1]])
        phase = int(np.angle(g1+s1*1j, deg=True))
        if phase < 0:
            phase = int(phase + 360)
        # print(phase)
        # circle1 = plt.Circle((cm[0][0], cm[0][1]), 0.03, color=c[i])
        # ax.add_patch(circle1)
    plt.show()

auxbool = False
if auxbool:
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
    ph.phasor_circle(ax)
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
    ph.phasor_circle(ax)
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
    ph.phasor_circle(ax)
    for i in range(len(cm)):
        circle1 = plt.Circle((cm[i][0], cm[i][1]), 0.03, color=c2[i])
        ax.add_patch(circle1)
    plt.show()
