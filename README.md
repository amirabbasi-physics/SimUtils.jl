# SimUtils.jl

[![CI](https://github.com/amirabbasi-physics/SimUtils.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/amirabbasi-physics/SimUtils.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Docs](https://github.com/amirabbasi-physics/SimUtils.jl/actions/workflows/Documentation.yml/badge.svg?branch=main)](https://github.com/amirabbasi-physics/SimUtils.jl/actions/workflows/Documentation.yml?query=branch%3Amain)

`SimUtils.jl` is a compact Julia package for FFT-based time-series analysis on
uniform grids. It provides CPU and GPU-backed utilities for discrete Fourier
transforms, linear correlations, mean-squared displacement calculations, and a
few small preprocessing helpers for padding and tapering.

The package is aimed at simulation and analysis workflows where you want:

- consistent CPU and GPU APIs
- explicit control over uniform-grid Fourier conventions
- linear, not circular, correlations
- lightweight utilities without a large analysis framework

## Features

- `FT`, `iFT`: forward and inverse discrete Fourier transforms with explicit
  angular-frequency grids and origin-phase handling
- `FT_GPU`, `iFT_GPU`: GPU-backed Fourier transforms using `CUDA.jl`
- `Correlation`, `Correlation_GPU`: linear cross-correlation and
  autocorrelation via zero-padded FFT convolution
- `MSD`, `MSD_GPU`: mean-squared displacement from autocorrelation identities
- `pad_to_power_of_two`, `apply_taper!`, `apply_taper2!`: preprocessing helpers

## Installation

Until the package is registered, install it directly from GitHub:

```julia
using Pkg
Pkg.add(url="https://github.com/amirabbasi-physics/SimUtils.jl.git")
```

Then load it with:

```julia
using SimUtils
```

## Quick Start

### Fourier transform on a uniform time grid

```julia
using SimUtils

t = 0.1 .* collect(0:127)
x = cos.(2π .* 1.5 .* t) .+ 0.2 .* sin.(2π .* 0.5 .* t)

ω, X = FT(t, x)
t_rec, x_rec = iFT(ω, X; t0=t[1])
```

### Linear correlation and MSD

```julia
using SimUtils
using Random

a = randn(1024)
b = randn(1024)

caa = Correlation(a)
cab = Correlation(a, b; subtract_mean=true)
msd = MSD(cumsum(randn(1024)))
```

### GPU-backed Fourier transform

```julia
using SimUtils
using Random

t = 0.01 .* collect(0:4095)
x = sin.(2π .* 3 .* t)

ω, X = FT_GPU(t, x)
```

The GPU variants use CUDA for the FFT itself and return ordinary Julia arrays on
the CPU for convenient downstream processing.

## Conventions and Assumptions

- The time or frequency grid must be uniformly spaced.
- `FT` returns angular frequencies `ω`, not cyclic frequencies `f`.
- `Correlation` computes linear correlations by zero-padding internally.
- `MSD` assumes a one-dimensional position time series. Multi-component MSD can
  be built by applying it component-wise and summing the results.

## Documentation

The package documentation is prepared with `Documenter.jl` and is intended to
be published to:

`https://amirabbasi-physics.github.io/SimUtils.jl/`

## License

This project is distributed under the terms of the license in [LICENSE](LICENSE).
