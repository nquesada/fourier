Fourier
=======

The function fourier_approx in fourier.py allows to calculate the Fourier transform of a function y = f(x)
that has been sampled in a grid of points x. That is, it will return values (k, F(k)) such that

```math
F(k) = \frac{1}{\sqrt{2 \pi}} \int dx \ e^{i k x} f(x)
```
where k is a vector of equally spaced and ordered values in reciprocal space.

