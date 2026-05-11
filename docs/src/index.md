# SimUtils.jl

`SimUtils.jl` provides a focused collection of FFT-based analysis tools for
uniformly sampled time series. It is designed for simulation workflows where
you want a small, explicit API instead of a large data-analysis framework.

```@docs
SimUtils
```

## What the package covers

- forward and inverse discrete Fourier transforms on uniform grids
- linear autocorrelation and cross-correlation via zero-padded FFTs
- mean-squared displacement from autocorrelation identities
- lightweight preprocessing helpers for zero-padding and tapering
- GPU-backed Fourier, correlation, and MSD routines using `CUDA.jl`

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

## Design Notes

The package deliberately stays narrow:

- It assumes uniformly spaced time or frequency grids.
- It returns angular frequencies `ω`.
- It exposes separate CPU and GPU entry points rather than hiding backend
  selection behind configuration state.
- It returns regular Julia arrays to keep downstream use simple.

## Package Structure

- `FT`, `iFT`: CPU Fourier transforms
- `FT_GPU`, `iFT_GPU`: GPU-backed Fourier transforms
- `Correlation`, `Correlation_GPU`: linear correlation utilities
- `MSD`, `MSD_GPU`: mean-squared displacement helpers
- `pad_to_power_of_two`, `apply_taper!`, `apply_taper2!`: preprocessing helpers

See [Examples](examples.md) for concrete workflows and [API](api.md) for the
full function reference.
