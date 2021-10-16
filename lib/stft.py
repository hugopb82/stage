import numpy as np

def get_frames(n, fft_size, hop_size):
	return np.ceil((n - fft_size) / hop_size).astype(int) + 1

def stft(x, fft_size, hop_size, fs=44100):
	"""Compute the Short Time Fourier Transform (STFT)

	Parameters :
	------------
		x : array
			The audio signal
		fft_size : int > 0
			The size of the fft frame (= the size of the window)
		hop_size : int > 0
			Number of audio samples between adjacent STFT columns

	Returns :
	---------
		X : np.ndarray
			STFT of x
		f : 
			Array of sample frequencies
		t : 
			Array of segment times 
	"""

	n = len(x)
	mid_fft = fft_size // 2 + 1
	n_frames = get_frames(n, fft_size, hop_size)
	x = np.concatenate((x, np.zeros((n_frames - 1) * hop_size + fft_size - n)))
	
	win = np.hanning(fft_size)
	#win = np.hanning(fft_size + 1)[:-1]	# better reconstruction

	X = np.zeros((mid_fft, n_frames), dtype = complex)
	for k in range(n_frames):
		start = k * hop_size
		x_frame = x[start : start + fft_size] * win
		X[:, k] = np.fft.rfft(x_frame, fft_size)

	f = [k * fs / fft_size for k in range(mid_fft)]
	t = [k * hop_size / fs for k in range(n_frames)]

	return X, f, t


def istft(X, fft_size, hop_size, fs=44100):
	"""Compute the Inverse STFT (ISTFT)

	Parameters :
	------------
		X : np.ndarray
			STFT of the audio signal
		fft_size : int > 0
			The size of the fft frame (= the size of the window)
		hop_size : int > 0
			Number of audio samples between adjacent STFT columns

	Returns :
	---------
		x : np.ndarray
			Reconstructed time domain signal
		t : 
			Array of segment times 
	"""

	fft_size = (X.shape[0] - 1) * 2
	n_frames = X.shape[1]

	win = np.hanning(fft_size)
	#win = np.hanning(fft_size + 1)[:-1]	 # better reconstruction

	n = (n_frames - 1) * hop_size + fft_size
	x = np.zeros(n, dtype = float)
	win_sum = np.zeros(n, dtype = float)
	for k in range(n_frames):
		start = k * hop_size
		x_frame = np.fft.irfft(X[:, k])
		x[start : start + fft_size] += x_frame * win
		win_sum[start : start + fft_size] += win ** 2

	win_sum[win_sum == 0] = np.finfo(np.float32).eps
	x /= win_sum

	t = [k / fs for k in range(n)]

	return x, t