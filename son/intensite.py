import librosa
import soundfile as sf
import matplotlib.pyplot as plt
import numpy as np

# Plot params
plt.rcParams["font.family"] = "Roboto"
plt.rcParams["figure.autolayout"] = True
plt.rcParams["axes.labelsize"] = 16
plt.rcParams["axes.titlesize"] = 24

# y, Fs = librosa.load('stage2/sounds/violin.wav')
# y = y[3*Fs:round(3.01*Fs)]
Fs = 22050
t = np.array([k / Fs for k in range(round(0.01*Fs))])
y1 = 0.5 * np.random.randn(round(0.01*Fs))
y2 = 2 * np.random.randn(round(0.01*Fs))


fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)

ax1.set_title("Un son doux")
ax1.set_xlabel('Temps [sec]')
ax1.set_ylabel('Pression')
ax1.plot(t, y1)

ax2.set_title("Un son fort")
ax2.set_xlabel('Temps [sec]')
#ax2.set_ylabel('Pression')
ax2.plot(t, y2)

fig.savefig('stage2/son/figures/intensite.png', transparent=True, dpi=300)
plt.show()