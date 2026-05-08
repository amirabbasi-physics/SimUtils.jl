function expected_omega_grid(n::Integer, dt::Real)
    return collect(FFTW.fftshift(2π .* FFTW.fftfreq(n, inv(dt))))
end

function expected_ft(t::AbstractVector{<:Real}, x::AbstractVector{<:Number})
    n = length(t)
    @assert length(x) == n
    dt = (t[end] - t[1]) / (n - 1)
    ω = expected_omega_grid(n, dt)
    X = collect(FFTW.fftshift(fft(x)) .* dt .* exp.(-im .* ω .* t[1]))
    return ω, X
end

function direct_correlation(a::AbstractVector{<:Real}, b::AbstractVector{<:Real}=a; subtract_mean::Bool=false)
    @assert length(a) == length(b)
    af = Float64.(a)
    bf = Float64.(b)
    if subtract_mean
        af .-= mean(af)
        bf .-= mean(bf)
    end

    n = length(af)
    c = Vector{Float64}(undef, n)
    for lag in 0:(n - 1)
        c[lag + 1] = sum(@view(af[1:(n - lag)]) .* @view(bf[(1 + lag):n])) / (n - lag)
    end
    return c
end

function direct_msd(positions::AbstractVector{<:Real})
    p = Float64.(positions)
    n = length(p)
    msd = Vector{Float64}(undef, n)
    for lag in 0:(n - 1)
        δ = @view(p[(1 + lag):n]) .- @view(p[1:(n - lag)])
        msd[lag + 1] = mean(abs2.(δ))
    end
    return msd
end
