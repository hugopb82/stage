import numpy as np
from scipy.io import wavfile
from scipy.ndimage.filters import maximum_filter, minimum_filter, uniform_filter

from stage2.lib.stft import stft, istft

# Load a .wav file
filename = 'billie'
sr, y = wavfile.read('stage2/sounds/' + filename + '.wav')

# Compute the STFT
fft_size = 4096
hop_size = fft_size // 4
Y, _, _ = stft(y, fft_size, hop_size)
S = np.abs(Y)

# Compute the FT2D
Y2 = np.fft.fftshift(np.fft.fft2(S))

# Compute the FT2D masks
neighborhood_size = (1, 35)

data = np.abs(Y2)
threshold = np.std(data)

data_max = maximum_filter(data, neighborhood_size)
maxima = (data == data_max)
data_min = minimum_filter(data, neighborhood_size)
diff = ((data_max - data_min) > threshold)
maxima[diff == 0] = 0

m_bg = maxima
m_fg = 1 - maxima

# Result spectrograms
x_bg = np.fft.ifft2(np.fft.ifftshift(m_bg * Y2))
x_fg = np.fft.ifft2(np.fft.ifftshift(m_fg * Y2))

# Spectrogram masks
m_bg = abs(x_bg) > abs(x_fg)
m_fg = 1 - m_bg

# Separation
Y_bg = m_bg * Y
Y_fg = m_fg * Y

# ISTFT
y_bg, _ = istft(Y_bg, fft_size, hop_size)
y_fg, _ = istft(Y_fg, fft_size, hop_size)

# Save
wavfile.write('stage2/2dft/results/' + filename + '_bg.wav', sr, y_bg.astype(np.int16))
wavfile.write('stage2/2dft/results/' + filename + '_fg.wav', sr, y_fg.astype(np.int16))