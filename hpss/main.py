import numpy as np
from scipy.io import wavfile
from scipy.ndimage import median_filter
from stage2.lib.stft import stft, istft

# Load a .wav file
filename = 'billie'
sr, y = wavfile.read('stage2/sounds/' + filename + '.wav')

# Compute the STFT
fft_size = 1024
hop_size = fft_size // 4
Y, _, _ = stft(y, fft_size, hop_size)

magnitude = np.abs(Y)

# Horizontal and vertical median filtering
win_h = 31
H = median_filter(magnitude, size=(1, win_h), mode="reflect")

win_p = 31
P = median_filter(magnitude, size=(win_p, 1), mode="reflect")

# Create binary masks
# M_h = H >= P
# M_p = P > H

# Soft masks
p = 2
a = H**p
b = P**p
c = a + b
M_h = a / c
M_p = b / c

# Spectrogram separation
Y_h = M_h * Y
Y_p = M_p * Y

# Compute the ISTFT
y_h, _ = istft(Y_h, fft_size, hop_size)
y_p, _ = istft(Y_p, fft_size, hop_size)

# Save
wavfile.write('stage2/hpss/results/' + filename + '_h.wav', sr, y_h.astype(np.int16))
wavfile.write('stage2/hpss/results/' + filename + '_p.wav', sr, y_p.astype(np.int16))