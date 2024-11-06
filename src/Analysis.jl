export FT, FT_GPU, iFT, iFT_GPU, MSD, MSD_GPU

function FT(t::Vector{T}, x::Union{Vector{T},Vector{Complex{T}}}, dt_dk_tolerance::T=1.0e-8,indvar::Bool=true) where T

    """
    Discrete fast Fourier transform.

    Takes the time series `t` and the function values `x` as arguments.
    By default, returns the FT and the frequency: setting `indvar=false` means the function returns only the FT.
    """
    a, b = minimum(t), maximum(t)
    dt = (t[end] - t[1]) / (length(t) - 1)  # timestep
    
    # Check if the time series is equally spaced
    if any(abs.(diff(t) .- dt) .> dt_dk_tolerance)
        error("Time series not equally spaced!")
    end

    N = length(t)
    # Calculate frequency values for FT
    k = fftshift(fftfreq(N, 1 ./ dt) .* 2 .* π)
    # Calculate FT of data
    xf = fftshift(fft(x))
    xf2 = xf .* (b - a) / N .* exp.(-1im .* k .* a)

    if indvar
        return collect(k), collect(xf2)
    else
        return collect(xf2)
    end
end

function FT_GPU(t::Vector{T}, x::Union{Vector{T},Vector{Complex{T}}}, dt_dk_tolerance::T=1.0e-8,indvar::Bool=true) where T
    """
    Discrete fast Fourier transform performed on GPU.

    Takes the time series `t` and the function values `x` as arguments.
    By default, returns the FT and the frequency: setting `indvar=false` means the function returns only the FT.
    """
    
    a, b = minimum(t), maximum(t)
    dt = (t[end] - t[1]) / (length(t) - 1)  # timestep

    # Check if the time series is equally spaced
    if any(abs.(diff(t) .- dt) .> dt_dk_tolerance)
        error("Time series not equally spaced!")
    end

    # Transfer data to GPU
    t_gpu = CuArray(t)
    x_gpu = CuArray(x)

    N = length(t_gpu)
    # Calculate frequency values for FT
    k = fftshift(fftfreq(N, 1 ./ dt) .* 2 .* π)
    # Calculate FT of data
    xf = fftshift(fft(x_gpu))
    xf2 = xf .* (b - a) / N .* exp.(-1im .* k .* a)

    if indvar
        return collect(k), collect(xf2)  # Convert back to CPU arrays for return
    else
        return collect(xf2)
    end
end

function iFT(k::Vector{T}, xf::Union{Vector{T},Vector{Complex{T}}}, dt_dk_tolerance::T=1.0e-8, indvar::Bool=true) where T
    """
    Inverse discrete fast Fourier transform.

    Takes the frequency series `k` and the function values `xf` as arguments.
    By default, returns the iFT and the time series; setting `indvar=false` means the function returns only the iFT.
    """
    dk = (k[end] - k[1]) / (length(k) - 1)  # timestep
    
    # Check if the frequency series is equally spaced
    if any(abs.(diff(k) .- dk) .> dt_dk_tolerance)
        error("Frequency series not equally spaced!")
    end

    N = length(k)
    x = ifftshift(ifft(xf))
    t = ifftshift(fftfreq(N, 1 ./ dk)) * 2 * π
    if N % 2 == 0
        x .*= exp.(-1im .* t .* N .* dk / 2) .* N .* dk / (2 * π)
    else
        x .*= exp.(-1im .* t .* (N - 1) .* dk / 2) .* N .* dk / (2 * π)
    end

    if indvar
        return collect(t), x
    else
        return x
    end
end

function iFT_GPU(k::Vector{T}, xf::Union{Vector{T},Vector{Complex{T}}}, dt_dk_tolerance::T=1.0e-8, indvar::Bool=true) where T
    """
    Inverse discrete fast Fourier transform performed on GPU.

    Takes the frequency series `k` and the function values `xf` as arguments.
    By default, returns the iFT and the time series; setting `indvar=false` means the function returns only the iFT.
    """
   
    
    dk = (k[end] - k[1]) / (length(k) - 1)  # timestep
    
    # Check if the frequency series is equally spaced
    if any(abs.(diff(k) .- dk) .> dt_dk_tolerance)
        error("Frequency series not equally spaced!")
    end

    # Transfer data to GPU
    k_gpu = CuArray(k)
    xf_gpu = CuArray(xf)

    N = length(k_gpu)
    x = ifftshift(ifft(xf_gpu))
    t = ifftshift(fftfreq(N, 1 ./ dk)) * 2 * π
    t_gpu = CuArray(t)

    if N % 2 == 0
        x .*= exp.(-1im .* t_gpu .* N .* dk ./ 2) .* N .* dk ./ (2 * π)
    else
        x .*= exp.(-1im .* t_gpu .* (N - 1) .* dk ./ 2) .* N .* dk ./ (2 * π)
    end

    if indvar
        return collect(t_gpu), collect(x)  # Convert back to CPU arrays for return
    else
        return collect(x)
    end
end





function Correlation(a::Vector{T}, b::Vector{T}=a, subtract_mean::Bool=false) where T
    """
    Calculate correlation or autocorrelation using fast Fourier transforms.

    If two arrays are given, calculates the correlation; if one array is given, calculates the autocorrelation.
    Setting `subtract_mean=true` causes the mean to be subtracted from the input data.
    """
    meana = subtract_mean ? mean(a) : zero(T)
    N = length(a)
    next_pow2 = 2 ^ nextpow(2, N)
    a2 = [a .- meana; zeros(T, next_pow2 - N)]
    data_a = [a2; zeros(T, length(a2))]

    fra = fft(data_a)

    if b === a
        sf = conj(fra) .* fra
    else
        meanb = subtract_mean ? mean(b) : zero(T)
        b2 = [b .- meanb; zeros(T, next_pow2 - length(b))]
        data_b = [b2; zeros(T, length(b2))]
        frb = fft(data_b)
        sf = conj(fra) .* frb
    end

    cor = real(ifft(sf))[1:N] ./ (N:-1:1)
    return cor
end
function MSD_GPU(positions::Vector{T}) where T 
    corr = Correlation_GPU(positions)
    return println(length(corr))
end

function MSD(positions::Vector{T}) where T 
    corr = Correlation(positions)
    # 2 .* (corr[1] .- corr[:])
    return println(length(corr))
end

function Correlation_GPU(a::Vector{T}, b::Vector{T}=a, subtract_mean::Bool=false) where T
    """
    Calculate correlation or autocorrelation using fast Fourier transforms on GPU.

    If two arrays are given, calculates the correlation; if one array is given, calculates the autocorrelation.
    Setting `subtract_mean=true` causes the mean to be subtracted from the input data.
    """
    meana = subtract_mean ? mean(a) : zero(T)
    N = length(a)
    next_pow2 = 2 ^ nextpow(2, N)
    a2 = [a .- meana; zeros(T, next_pow2 - N)]
    data_a = [a2; zeros(T, length(a2))]

    data_a_gpu = CuArray(data_a)
    fra = fft(data_a_gpu)

    if b === a
        sf = conj(fra) .* fra
    else
        meanb = subtract_mean ? mean(b) : zero(T)
        b2 = [b .- meanb; zeros(T, next_pow2 - length(b))]
        data_b = [b2; zeros(T, length(b2))]
        data_b_gpu = CuArray(data_b)
        frb = fft(data_b_gpu)
        sf = conj(fra) .* frb
    end

    isf_gpu = ifft(sf)
    cor = real(Array(isf_gpu))[1:N] ./ (N:-1:1)
    return collect(cor)  # Move the result back to CPU
end

export apply_taper!, apply_taper2!

function apply_taper!(data, taper_start, taper_end)
    for i in taper_start:taper_end
        data[i] *= 0.5 * (1 - cos(π * (i - taper_start) / (taper_end - taper_start)))
    end
end

function apply_taper2!(data, taper_start, taper_end)
    for i in taper_start:taper_end
        data[i] *= cos(π * (i - taper_start) / (2 * (taper_end - taper_start)))
    end
    return nothing
end
	

export correlation

function correlation(a::Vector{T}, b::Vector{T} = a; subtract_mean::Bool=false) where T
    
    function pad_to_power_of_two(x, subtract_mean_val)
        n = length(x)
        next_power_of_two = 2^ceil(Int, log2(n))
        padded_length = next_power_of_two - n
        append!(x .- subtract_mean_val, zeros(padded_length))
    end

    meana = subtract_mean ? mean(a) : 0.0
    a2 = pad_to_power_of_two(a, meana)
    data_a = append!(a2, zeros(length(a2)))
    fra = fft(data_a)

    if b === nothing
        sf = conj.(fra) .* fra
    else
        meanb = subtract_mean ? mean(b) : 0.0
        b2 = pad_to_power_of_two(b, meanb)
        data_b = append!(b2, zeros(length(b2)))
        frb = fft(data_b)
        sf = conj.(fra) .* frb
    end

    cor = real(ifft(sf)[1:length(a)]) ./ reverse(1:length(a))
    return cor #, sf[1:length(a)]
end
	
export pad_to_power_of_two
	
function pad_to_power_of_two(x::Vector{T}, subtract_mean_val::T) where T
    n = length(x)
    next_power_of_two = 2^ceil(Int, log2(n))
    padded_length = next_power_of_two - n
    padded_x = Vector{T}(undef, next_power_of_two)
    padded_x[1:n] .= x .- subtract_mean_val
    padded_x[n+1:end] .= zero(T)
    return padded_x
end

export correlation_GPU

function correlation_GPU(a::Vector{T}, b::Vector{T} = a; subtract_mean::Bool=false) where T
    

    meana = subtract_mean ? mean(a) : zero(T)
    a2 = pad_to_power_of_two(a, meana)
    data_a = vcat(a2, zeros(eltype(a2), length(a2)))
    data_a_gpu = CuVector(data_a)
    fra = fft(data_a_gpu)

    if b === nothing
        sf = conj.(fra) .* fra
    else
        meanb = subtract_mean ? mean(b) : zero(T)
        b2 = pad_to_power_of_two(b, meanb)
        data_b = vcat(b2, zeros(eltype(b2), length(b2)))
        data_b_gpu = CuVector(data_b)
        frb = fft(data_b_gpu)
        sf = conj.(fra) .* frb
    end

    #cor = real(ifft(sf)[1:length(a_gpu)]) ./ reverse(CuArray(1:length(a_gpu)))
    isf_gpu = ifft(sf)
    cor = real(Array(isf_gpu))[1:length(a)] ./ reverse(1:length(a))
    return Array(cor)  # Move result back to CPU
end

