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
y = y[3*Fs:round(3.01*Fs)]
t = [k / Fs for k in range(round(0.01*Fs))]

plt.figure()
plt.ylabel('Pression')
plt.xlabel('Temps [sec]')
plt.plot(t, y)
plt.savefig('stage2/son/figures/waveform.png', transparent=True, dpi=300)
plt.show()