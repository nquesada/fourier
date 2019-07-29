import pytest
import numpy as np
from fourier import fourier_approx


def norm_gaussian(x, mu=1.0, k0=0.0):
    return np.exp(-0.5 * (x * mu) ** 2) / np.sqrt(np.sqrt(np.pi) / mu) * np.exp(1j * k0 * x)


@pytest.mark.parametrize("N", [400, 401])
@pytest.mark.parametrize("mu", [0.5, 1, 2])
@pytest.mark.parametrize("k0", [-1, 0, 1])
@pytest.mark.parametrize("x0", [-1, 0, 1])
def test_gaussian(mu, k0, x0, N):
    xf = 40
    x = np.linspace(-xf, xf, N)
    dx = x[1] - x[0]
    y = norm_gaussian(x - x0, mu=mu, k0=k0)
    xt, yt = fourier_approx(x, y)
    exact = np.exp(-1j * k0 * x0) * norm_gaussian(xt - k0, mu=1.0 / mu, k0=-x0)
    assert np.allclose(yt, exact)


@pytest.mark.parametrize("N", [400, 401])
@pytest.mark.parametrize("mu", [0.5, 1, 2])
@pytest.mark.parametrize("k0", [-1, 0, 1])
@pytest.mark.parametrize("x0", [-1, 0, 1])
def test_gaussian(mu, k0, x0, N):
    xf = 40
    x = np.linspace(-xf, xf, N)
    dx = x[1] - x[0]
    y = norm_gaussian(x - x0, mu=mu, k0=k0)
    xt, yt = fourier_approx(x, y)
    exact = np.exp(-1j * k0 * x0) * norm_gaussian(xt - k0, mu=1.0 / mu, k0=-x0)

    assert np.allclose(yt, exact)
