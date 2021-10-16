from scipy.io import wavfile
import numpy as np						# Scientific calculations
import matplotlib.pyplot as plt			# Plotting
plt.style.use('stage2/perso.mplstyle')
from scipy.ndimage.filters import maximum_filter, minimum_filter, uniform_filter
from stage2.lib.stft import stft, istft

# Load a sound
filename = 'billie'
sr, y = wavfile.read('stage2/sounds/' + filename + '.wav')

# Compute the STFT
fft_size = 4096
hop_size = fft_size // 4

Y, f, t = stft(y, fft_size, hop_size)
S = np.abs(Y)

plt.figure("Spectrogramme")
plt.imshow(np.log10(S), origin='lower', extent=[0,t[-1],0,f[-1]], aspect='auto')
plt.ylabel('Fréquence [Hz]')
plt.xlabel('Temps [sec]')
plt.savefig('stage2/2dft/figures/spectrogramme.png', transparent=True, dpi=300)
#plt.show()

# Compute the 2DFT
Y2 = np.fft.fftshift(np.fft.fft2(S))

plt.figure('TF2D du module du spectrogramme')
plt.imshow(np.log10(np.abs(Y2)), origin='lower', aspect='auto')
plt.savefig('stage2/2dft/figures/2dft.png', transparent=True, dpi=300)
#plt.show()

# Find peaks in the 2DFT
neighborhood_size = (1, 35)

data = np.abs(Y2)
threshold = np.std(data)

data_max = maximum_filter(data, neighborhood_size)
maxima = (data == data_max)
data_min = minimum_filter(data, neighborhood_size)
diff = ((data_max - data_min) > threshold)
maxima[diff == 0] = 0


# Create bg / fg masks for the 2DFT
M_bg = maxima
M_fg = 1 - maxima

# Separate the 2DFT
Y2_bg = M_bg * Y2
Y2_fg = M_fg * Y2

plt.figure('Séparation de la 2DFT')

plt.subplot(1, 2, 1)
plt.imshow(np.log10(np.abs(Y2_bg)), origin='lower', aspect='auto')
plt.title('Arrière-plan')

plt.subplot(1, 2, 2)
plt.imshow(np.log10(np.abs(Y2_fg)), origin='lower', aspect='auto')
plt.title('Avant-plan')

plt.tight_layout()
plt.savefig('stage2/2dft/figures/masked_2dft.png', transparent=True, dpi=300)
# plt.show()

# Result spectrograms
Y_bg = np.fft.ifft2(np.fft.ifftshift(Y2_bg))
Y_fg = np.fft.ifft2(np.fft.ifftshift(Y2_fg))

plt.figure('Spectrogrammes séparés', figsize=(12,6))

plt.subplot(1, 2, 1)
plt.imshow(np.log10(np.abs(Y_bg)), origin='lower', extent=[0,t[-1],0,f[-1]], aspect='auto')
plt.title('Spectrogramme de l\'arrière plan')

plt.subplot(1, 2, 2)
plt.imshow(np.log10(np.abs(Y_fg)), origin='lower', extent=[0,t[-1],0,f[-1]], aspect='auto')
plt.title('Spectrogramme de l\'avant plan')

plt.savefig('stage2/2dft/figures/separated_spectrograms.png', transparent=True, dpi=300)
# plt.show()

# Create bg / fg masks for the STFT
m_bg = abs(Y_bg) > abs(Y_fg)
# m_bg[1:10][:] = 1
m_fg = 1 - m_bg

# Separate the STFT
Y_bg = m_bg * Y
Y_fg = m_fg * Y

plt.figure('Spectrogrammes séparés', figsize=(12,6))

plt.subplot(1, 2, 1)
plt.imshow(np.log10(np.abs(Y_bg)), origin='lower', extent=[0,t[-1],0,f[-1]], aspect='auto')
plt.title('Spectrogramme de l\'arrière plan')

plt.subplot(1, 2, 2)
plt.imshow(np.log10(np.abs(Y_fg)), origin='lower', extent=[0,t[-1],0,f[-1]], aspect='auto')
plt.title('Spectrogramme de l\'avant plan')

plt.savefig('stage2/2dft/figures/separated_spectrograms_2.png', transparent=True, dpi=300)
plt.show()

# ISTFT
y_bg, _ = istft(Y_bg, fft_size, hop_size)
y_fg, _ = istft(Y_fg, fft_size, hop_size)

# Save
wavfile.write('stage2/2dft/results/' + filename + '_bg.wav', sr, y_bg.astype(np.int16))
wavfile.write('stage2/2dft/results/' + filename + '_fg.wav', sr, y_fg.astype(np.int16))