import numpy as np


def fourier_approx(x, y, scale=True):
    r"""Calculates the fourier transform of the function y = f(x) where x is 
        a vector of equally spaced values. The fourier transform follows the conventions 
        of numpy:
        
        F(k) = \frac{1}{\sqrt{2 \pi}} \int dx e^{i k x} f(x)
        
        and also is unitary meaning that
        
        \sum |F(k)|^2 dk \approx \int |F(k)|^2 dk =  \int |f(x)|^2 dx \approx \sum |f(x)|^2 dx
        Args:
              x (array): vector of abscissae, grid the function f(x) is evaluated
              y (array): vector of ordinates, y = f(x)
              scale (bool): Makes the transform unitary with respect to integration as above.
                            Otherwise it guarantees that \sum |F(k)|^2 = \sum |f(x)|^2 
        
        Returns:
             tuple (k, F(k)): The abcissae (k) and ordinate F(k) of the fourier transform of y = f(x)
        """
    N = len(x)
    assert N == len(y)
    dx = x[1] - x[0]
    assert np.allclose(np.diff(x), dx * np.ones(N - 1))
    yt = np.fft.fftshift(np.fft.fft(y, norm="ortho"))
    xt = np.fft.fftshift(np.fft.fftfreq(len(x), dx)) * 2 * np.pi
    if N % 2 == 0:
        s = 0.0
    else:
        s = 0.5
    phases = np.exp(-1j * np.pi * (np.arange(N) - 0.5 + (np.arange(N) + s) / N))
    yt = phases * yt
    if scale:
        dk = xt[1] - xt[0]
        yt = np.sqrt(dx / dk) * yt
    return xt, yt
