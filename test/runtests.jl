using Pkg
Pkg.develop(path="/net/storage/abbaa90/.julia/dev/MyUtilities")
Pkg.instantiate()
using MyUtilities

using Test
using FFTW
using Statistics
using CUDA

const HAS_CUDA = CUDA.functional()

include("reference_helpers.jl")

@testset "MyUtilities.jl" begin
    include("pad_and_taper_tests.jl")
    include("fourier_tests.jl")
    include("correlation_tests.jl")
    include("msd_tests.jl")
    include("gpu_tests.jl")
end
