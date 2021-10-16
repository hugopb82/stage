import librosa
import soundfile as sf
import matplotlib.pyplot as plt
import numpy as np

# Plot params
plt.rcParams["font.family"] = "Roboto"
plt.rcParams["figure.autolayout"] = True
plt.rcParams["axes.labelsize"] = 16
plt.rcParams["axes.titlesize"] = 24

y, Fs = librosa.load('stage2/sounds/violin.wav')
y1 = y[3*Fs:round(3.01*Fs)]
t = np.array([k / Fs for k in range(round(0.01*Fs))])
y2 = 0.5 * np.random.randn(round(0.01*Fs))


fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)

ax1.set_title("Violon")
ax1.set_xlabel('Temps [sec]')
ax1.set_ylabel('Pression')
ax1.plot(t, y1)

ax2.set_title("Marteau-piqueur")
ax2.set_xlabel('Temps [sec]')
#ax2.set_ylabel('Pression')
ax2.plot(t, y2)

fig.savefig('stage2/son/figures/vlvsmp.png', transparent=True, dpi=300)
plt.show()