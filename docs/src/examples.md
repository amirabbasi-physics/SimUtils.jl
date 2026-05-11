# Examples

## Fourier transform and exact inverse on a shifted grid

```julia
using SimUtils

t0 = -0.5
t = t0 .+ 0.05 .* collect(0:255)
x = cos.(2π .* 1.2 .* t) .+ 0.1 .* sin.(2π .* 0.4 .* t)

ω, X = FT(t, x)
t_rec, x_rec = iFT(ω, X; t0=t[1])
```

`iFT(...; t0=t[1])` compensates for the phase convention used in `FT`, so the
reconstructed signal matches the original sampling grid.

## Linear cross-correlation

```julia
using SimUtils
using Random

a = randn(4096)
b = randn(4096)

c_ab = Correlation(a, b; subtract_mean=true)
c_aa = Correlation(a)
```

The implementation uses zero-padding internally, so the result is a linear
correlation instead of the circular correlation returned by a raw FFT product.

## Mean-squared displacement

```julia
using SimUtils
using Random

positions = cumsum(randn(2048))
msd = MSD(positions)
```

For a one-dimensional trajectory, `MSD` uses the autocorrelation identity

`MSD(τ) = 2(C(0) - C(τ))`

but keeps the finite-sample normalization explicit.

## GPU-backed analysis

```julia
using SimUtils
using Random

t = 0.01 .* collect(0:8191)
x = sin.(2π .* 2 .* t)

ω, X = FT_GPU(t, x)
msd = MSD_GPU(cumsum(randn(4096)))
```

The GPU variants use CUDA for the heavy FFT step and return the final result on
the CPU.
