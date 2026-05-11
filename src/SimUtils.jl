"""
    SimUtils

Utilities for uniformly sampled time-series analysis, with CPU and GPU-backed
implementations for:

- discrete Fourier transforms on uniform grids
- linear correlations via FFT-based convolution
- mean-squared displacement from autocorrelation identities
- lightweight tapering and zero-padding helpers

The exported GPU variants use `CUDA.jl` for the FFT step and return results on
the CPU for convenient downstream analysis.
"""
module SimUtils

include("Analysis.jl")

end
