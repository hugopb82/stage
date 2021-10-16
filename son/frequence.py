import librosa
import soundfile as sf
import matplotlib.pyplot as plt
import numpy as np

# Plot params
plt.rcParams["font.family"] = "Roboto"
plt.rcParams["figure.autolayout"] = True
plt.rcParams["axes.labelsize"] = 16
plt.rcParams["axes.titlesize"] = 24

Fs = 22050
t = np.array([k / Fs for k in range(round(0.01*Fs))])
y1 = np.cos(200 * 2 * np.pi * t)
y2 = np.cos(1000 * 2 * np.pi * t)

fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)

ax1.set_title("Un son grave")
ax1.set_xlabel('Temps [sec]')
ax1.set_ylabel('Pression')
ax1.plot(t, y1)

ax2.set_title("Un son aigu")
ax2.set_xlabel('Temps [sec]')
#ax2.set_ylabel('Pression')
ax2.plot(t, y2)

fig.savefig('stage2/son/figures/frequence.png', transparent=True, dpi=300)
plt.show()