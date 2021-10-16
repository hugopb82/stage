import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import median_filter
from scipy.io import wavfile

from stage2.lib.stft import stft, istft

# Plot params
plt.rcParams["font.family"] = "Roboto"
plt.rcParams["figure.autolayout"] = True
plt.rc('axes', labelsize=16)
plt.rc('axes', titlesize=24)

# Params
fft_size = 1024
hop_size = fft_size // 4

# Harmonic vs Percussive sounds
sr, y_h = wavfile.read('stage2/sounds/billie.wav')
sr, y_p = wavfile.read('stage2/sounds/billie.wav')

Y_h, f, t = stft(y_h, fft_size, hop_size, sr)
Y_p, f_p, t_p = stft(y_p, fft_size, hop_size, sr)

S_h = 20 * np.log(abs(Y_h))
S_p = 20 * np.log(abs(Y_p))

plt.figure()

plt.subplot(1, 2, 1)
plt.imshow(S_h, origin='lower', extent=[0,t[-1],0,f[-1]], aspect='auto')
plt.title('Violon')
plt.ylabel('Fréquence [Hz]')
plt.xlabel('Temps [sec]')
# plt.yscale('log')

plt.subplot(1, 2, 2)
plt.imshow(S_p, origin='lower', extent=[0,t_p[-1],0,f_p[-1]], aspect='auto')
plt.title('Batterie')
plt.ylabel('Fréquence [Hz]')
plt.xlabel('Temps [sec]')
# plt.yscale('log')

plt.savefig('stage2/hpss/figures/spectrogrammes.png', transparent=True, dpi=300)
plt.show()

### Superposition
Y = Y_h + Y_p
Y = Y_h
S = 20 * np.log(np.abs(Y))

plt.figure()
plt.imshow(S, origin='lower', extent=[0,t[-1],0,f[-1]], aspect='auto')
plt.title('Violon + Batterie')
plt.ylabel('Fréquence [Hz]')
plt.xlabel('Temps [sec]')
plt.savefig('stage2/hpss/figures/superposition.png', transparent=True, dpi=300)
plt.show()

### Filtering
win_h = 17
H = median_filter(S, size=(1, win_h), mode="reflect")

win_p = 17
P = median_filter(S, size=(win_p, 1), mode="reflect")

plt.figure()

plt.subplot(1, 2, 1)
plt.imshow(H, origin='lower', extent=[0,t[-1],0,f[-1]], aspect='auto')
plt.title('Filtrage horizontal')
plt.ylabel('Fréquence [Hz]')
plt.xlabel('Temps [sec]')
# plt.yscale('log')

plt.subplot(1, 2, 2)
plt.imshow(P, origin='lower', extent=[0,t_p[-1],0,f_p[-1]], aspect='auto')
plt.title('Filtrage vertical')
plt.ylabel('Fréquence [Hz]')
plt.xlabel('Temps [sec]')
# plt.yscale('log')

plt.savefig('stage2/hpss/figures/filtrages.png', transparent=True, dpi=300)
plt.show()

### Masking
M_h = H >= P
M_p = P > H

plt.figure()

plt.subplot(1, 2, 1)
plt.imshow(M_h, origin='lower', extent=[0,t[-1],0,f[-1]], aspect='auto')
plt.title('Masque harmonique')
plt.ylabel('Fréquence [Hz]')
plt.xlabel('Temps [sec]')
# plt.yscale('log')

plt.subplot(1, 2, 2)
plt.imshow(M_p, origin='lower', extent=[0,t_p[-1],0,f_p[-1]], aspect='auto')
plt.title('Masque percussif')
plt.ylabel('Fréquence [Hz]')
plt.xlabel('Temps [sec]')
# plt.yscale('log')

plt.savefig('stage2/hpss/figures/masques.png', transparent=True, dpi=300)
plt.show()

### Separation
Y_h = M_h * Y
Y_p = M_p * Y

S_h = 20 * np.log(np.abs(Y_h))
S_p = 20 * np.log(np.abs(Y_p))

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.imshow(S_h, origin='lower', extent=[0,t[-1],0,f[-1]], aspect='auto')
plt.title('Composante harmonique')
plt.ylabel('Fréquence [Hz]')
plt.xlabel('Temps [sec]')
# plt.yscale('log')

plt.subplot(1, 2, 2)
plt.imshow(S_p, origin='lower', extent=[0,t_p[-1],0,f_p[-1]], aspect='auto')
plt.title('Composante percussive')
plt.ylabel('Fréquence [Hz]')
plt.xlabel('Temps [sec]')
# plt.yscale('log')

plt.savefig('stage2/hpss/figures/resultat.png', transparent=True, dpi=300)
plt.show()

# ISTFT
y_h, _ = istft(Y_h, fft_size, hop_size)
y_p, _ = istft(Y_p, fft_size, hop_size)

# Save
wavfile.write('stage2/hpss/results/harmonic.wav', sr, y_h.astype(np.int16))
wavfile.write('stage2/hpss/results/percussive.wav', sr, y_p.astype(np.int16))