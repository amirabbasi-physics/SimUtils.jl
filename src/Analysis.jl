###############
# Dependencies
###############
using FFTW
using CUDA
using Statistics

################
# Public exports
################
export FT, FT_GPU, iFT, iFT_GPU, MSD, MSD_GPU
export Correlation, Correlation_GPU
export apply_taper!, apply_taper2!
export correlation, correlation_GPU
export pad_to_power_of_two

########################
# Utility / helper stuff
########################

"""
    pad_to_power_of_two(x, meanval)

Return a copy of `x` zero-padded to the next power-of-two length after first
subtracting `meanval` from the populated portion.

This helper is used internally to make FFT-based correlation routines more
efficient while keeping the linear-correlation normalization explicit.
"""
function pad_to_power_of_two(x::AbstractVector{T}, meanval::Number) where {T}
    n = length(x)
    m = nextpow(2, n)                   # smallest power of 2 ≥ n
    S = promote_type(T, typeof(meanval))
    y = Vector{S}(undef, m)
    @inbounds begin
        @views y[1:n] .= x .- meanval
        @views y[n+1:m] .= zero(S)
    end
    return y
end

"""
    apply_taper!(data, taper_start, taper_end)

Apply a rising half-Hann taper in place on the inclusive index interval
`[taper_start, taper_end]`.
"""
function apply_taper!(data, taper_start::Integer, taper_end::Integer)
    taper_end > taper_start || throw(ArgumentError("taper_end must be greater than taper_start."))
    @inbounds for i in taper_start:taper_end
        data[i] *= 0.5 * (1 - cos(π * (i - taper_start) / (taper_end - taper_start)))
    end
    return nothing
end

"""
    apply_taper2!(data, taper_start, taper_end)

Apply a cosine half-window variant in place on the inclusive index interval
`[taper_start, taper_end]`.
"""
function apply_taper2!(data, taper_start::Integer, taper_end::Integer)
    taper_end > taper_start || throw(ArgumentError("taper_end must be greater than taper_start."))
    @inbounds for i in taper_start:taper_end
        data[i] *= cos(π * (i - taper_start) / (2 * (taper_end - taper_start)))
    end
    return nothing
end

############################
# Fourier transform routines
############################

"""
    FT(t, x; tol=1e-8, indvar=true)

Discrete FT returning angular frequencies `ω` and spectrum `X(ω)` on a uniformly spaced time grid `t`.
Scaling is such that `X = dt * FFT(x)` (up to fftshift), with an origin correction so you can use `iFT(...; t0=t[1])` to invert exactly.
"""
function FT(t::AbstractVector{T}, x::AbstractVector{<:Number}, tol::Real=1e-8, indvar::Bool=true) where {T<:Real}
    N = length(t)
    @assert length(x) == N "t and x must have the same length"
    @assert N ≥ 2 "Need at least 2 samples"

    dt = (t[end] - t[1]) / (N - 1)
    # uniform grid check
    if any(abs.(diff(t) .- dt) .> tol)
        error("Time series not equally spaced!")
    end

    ω = FFTW.fftshift(2π .* FFTW.fftfreq(N, inv(dt)))     # angular frequencies
    X = FFTW.fftshift(fft(x)) .* dt .* exp.(-im .* ω .* t[1])  # dt scaling + origin phase

    return indvar ? (collect(ω), collect(X)) : collect(X)
end

"""
    FT_GPU(t, x; tol=1e-8, indvar=true)

GPU version of `FT`. Only the FFT itself runs on the GPU; results are returned on the CPU.
"""
function FT_GPU(t::AbstractVector{T}, x::AbstractVector{<:Number}, tol::Real=1e-8, indvar::Bool=true) where {T<:Real}
    N = length(t)
    @assert length(x) == N "t and x must have the same length"
    @assert N ≥ 2 "Need at least 2 samples"

    dt = (t[end] - t[1]) / (N - 1)
    if any(abs.(diff(t) .- dt) .> tol)
        error("Time series not equally spaced!")
    end

    ω = FFTW.fftshift(2π .* FFTW.fftfreq(N, inv(dt)))     # on CPU
    x_gpu = CuArray(x)
    X_gpu = fft(x_gpu)                                     # cuFFT
    Xs_gpu = FFTW.fftshift(X_gpu)                          # shift after FFT
    # multiply by dt and origin phase (on GPU), then bring back
    phase_gpu = exp.(-im .* CuArray(ω) .* t[1])
    X = Array(Xs_gpu .* dt .* phase_gpu)

    return indvar ? (collect(ω), X) : X
end

"""
    iFT(ω, X; tol=1e-8, indvar=true, t0=0.0)

Inverse of `FT`. If you originally used `FT(t,x)` with time origin `t[1]`, call as `iFT(ω, X; t0=t[1])` to recover the original grid.
Returns `(t, x)` with `t = t0 .+ (0:N-1)*dt`, where `dt = 2π/(N Δω)`.
"""
function iFT(ω::AbstractVector{T}, X::AbstractVector{<:Number}, tol::Real=1e-8, indvar::Bool=true; t0::Real=0.0) where {T<:Real}
    N = length(ω)
    @assert length(X) == N "ω and X must have the same length"
    @assert N ≥ 2 "Need at least 2 samples"

    Δω = ω[2] - ω[1]
    if any(abs.(diff(ω) .- Δω) .> tol)
        error("Frequency grid not equally spaced!")
    end

    ωu = FFTW.ifftshift(ω)
    Xu = FFTW.ifftshift(X)

    dt = 2π / (N * Δω)                              # dual grid spacing
    # undo origin phase and invert scaling by dt
    x = (1/dt) .* ifft(Xu .* exp.(im .* ωu .* t0))
    t = t0 .+ dt .* collect(0:N-1)

    return indvar ? (collect(t), collect(x)) : collect(x)
end

"""
    iFT_GPU(ω, X; tol=1e-8, indvar=true, t0=0.0)

GPU version of `iFT`. Returns arrays on the CPU.
"""
function iFT_GPU(ω::AbstractVector{T}, X::AbstractVector{<:Number}, tol::Real=1e-8, indvar::Bool=true; t0::Real=0.0) where {T<:Real}
    N = length(ω)
    @assert length(X) == N "ω and X must have the same length"
    @assert N ≥ 2 "Need at least 2 samples"

    Δω = ω[2] - ω[1]
    if any(abs.(diff(ω) .- Δω) .> tol)
        error("Frequency grid not equally spaced!")
    end

    ωu = FFTW.ifftshift(ω)

    dt = 2π / (N * Δω)

    Xu_gpu = CuArray(FFTW.ifftshift(X))
    phase_gpu = exp.(im .* CuArray(ωu) .* t0)
    x_gpu = (1/dt) .* ifft(Xu_gpu .* phase_gpu)           # cuFFT inverse
    x = Array(x_gpu)
    t = t0 .+ dt .* collect(0:N-1)

    return indvar ? (collect(t), x) : x
end

##########################################
# Correlation / Autocorrelation (linear)
##########################################

"""
    Correlation(a[, b=a]; subtract_mean=false)

Linear (non-circular) correlation via FFT with zero padding and
normalization by the decreasing sample count `(N:-1:1)`.
If `b==a`, this is the autocorrelation.
"""
function Correlation(a::AbstractVector{T}, b::AbstractVector{T}=a; subtract_mean::Bool=false) where {T<:Real}
    N = length(a)
    @assert N ≥ 1
    @assert length(b) == N "a and b must have the same length"

    meana = subtract_mean ? mean(a) : zero(T)
    a2 = pad_to_power_of_two(a, meana)
    data_a = vcat(a2, zeros(T, length(a2)))               # zero-pad to linearize
    Fa = fft(data_a)

    S = if b === a
        conj.(Fa) .* Fa
    else
        meanb = subtract_mean ? mean(b) : zero(T)
        b2 = pad_to_power_of_two(b, meanb)
        data_b = vcat(b2, zeros(T, length(b2)))
        Fb = fft(data_b)
        conj.(Fa) .* Fb
    end

    c_full = real(ifft(S))
    c = @view c_full[1:N]
    return c ./ collect(N:-1:1)
end

"""
    Correlation_GPU(a[, b=a]; subtract_mean=false)

GPU version of `Correlation`. Returns result on CPU.
"""
function Correlation_GPU(a::AbstractVector{T}, b::AbstractVector{T}=a; subtract_mean::Bool=false) where {T<:Real}
    N = length(a)
    @assert N ≥ 1
    @assert length(b) == N "a and b must have the same length"

    meana = subtract_mean ? mean(a) : zero(T)
    a2 = pad_to_power_of_two(a, meana)
    data_a = vcat(a2, zeros(eltype(a2), length(a2)))
    Fa = fft(CuArray(data_a))

    S_gpu = if b === a
        conj.(Fa) .* Fa
    else
        meanb = subtract_mean ? mean(b) : zero(T)
        b2 = pad_to_power_of_two(b, meanb)
        data_b = vcat(b2, zeros(eltype(b2), length(b2)))
        Fb = fft(CuArray(data_b))
        conj.(Fa) .* Fb
    end

    c_full = real(Array(ifft(S_gpu)))
    c = @view c_full[1:N]
    return c ./ collect(N:-1:1)
end

######################
# MSD (1D time series)
######################

"""
    MSD(positions)

Mean-squared displacement computed from autocorrelation:
`MSD(τ) = 2 (C(0) - C(τ))`, where `C` is the autocorrelation of the position time series.
"""
function MSD(positions::AbstractVector{T}) where {T<:Real}
    N = length(positions)
    @assert N ≥ 1

    c = Correlation(positions)
    squares = abs2.(positions)
    prefix = cumsum(squares)
    total = prefix[end]

    msd = similar(c)
    @inbounds for lag in 0:(N - 1)
        count = N - lag
        sum_head = prefix[count]
        sum_tail = lag == 0 ? total : total - prefix[lag]
        msd[lag + 1] = (sum_head + sum_tail) / count - 2 * c[lag + 1]
    end

    return msd
end

"""
    MSD_GPU(positions)

GPU-backed autocorrelation; returns MSD on CPU.
"""
function MSD_GPU(positions::AbstractVector{T}) where {T<:Real}
    N = length(positions)
    @assert N ≥ 1

    c = Correlation_GPU(positions)
    squares = abs2.(positions)
    prefix = cumsum(squares)
    total = prefix[end]

    msd = similar(c)
    @inbounds for lag in 0:(N - 1)
        count = N - lag
        sum_head = prefix[count]
        sum_tail = lag == 0 ? total : total - prefix[lag]
        msd[lag + 1] = (sum_head + sum_tail) / count - 2 * c[lag + 1]
    end

    return msd
end

########################################
# Lowercase wrappers (back-compat names)
########################################

"""
    correlation(a[, b=a]; subtract_mean=false)

Lowercase alias for [`Correlation`](@ref).
"""
function correlation(a::AbstractVector{T}, b::AbstractVector{T}=a; subtract_mean::Bool=false) where {T<:Real}
    return Correlation(a, b; subtract_mean=subtract_mean)
end

"""
    correlation_GPU(a[, b=a]; subtract_mean=false)

Lowercase alias for [`Correlation_GPU`](@ref).
"""
function correlation_GPU(a::AbstractVector{T}, b::AbstractVector{T}=a; subtract_mean::Bool=false) where {T<:Real}
    return Correlation_GPU(a, b; subtract_mean=subtract_mean)
end
