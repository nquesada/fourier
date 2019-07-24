import numpy as np

def fourier_approx(x, y, scale = True):
	N = len(x)
	assert N == len(y)
	dx = x[1]-x[0]
	assert np.allclose(np.diff(x), dx*np.ones(N-1))
	yt = np.fft.fftshift(np.fft.fft(y,norm="ortho"))
	xt = np.fft.fftshift(np.fft.fftfreq(len(x),dx))*2*np.pi
	if N%2 ==0:
		s = 0.0
	else:
		s = 0.5
	phases = np.exp(-1j*np.pi*(np.arange(N)-0.5+(np.arange(N)+s)/N))
	yt = phases*yt
	if scale:
		dk = xt[1]-xt[0]
		yt = np.sqrt(dx/dk)*yt
	return xt, yt