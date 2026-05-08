@testset "GPU Variants" begin
    if !HAS_CUDA
        @test_skip "CUDA.functional() returned false"
    else
        t = 0.125 .* collect(0:7)
        x = sin.(2π .* t)

        ω_cpu, X_cpu = FT(t, x)
        ω_gpu, X_gpu = FT_GPU(t, x)
        @test ω_gpu ≈ ω_cpu
        @test X_gpu ≈ X_cpu atol=1e-10 rtol=1e-10

        ω_ref, X_ref = expected_ft(t, x)
        inverse_gpu_result = try
            iFT_GPU(ω_ref, X_ref; t0=t[1])
        catch err
            err
        end
        @test !(inverse_gpu_result isa Exception)
        if !(inverse_gpu_result isa Exception)
            t_gpu, x_gpu = inverse_gpu_result
            @test t_gpu ≈ t atol=1e-12 rtol=1e-12
            @test real.(x_gpu) ≈ x atol=1e-10 rtol=1e-10
            @test maximum(abs.(imag.(x_gpu))) ≤ 1e-10
        end

        a = [1.0, -1.0, 2.0, 0.5]
        b = [0.5, 3.0, -2.0, 1.0]
        @test Correlation_GPU(a, b) ≈ direct_correlation(a, b)
        @test correlation_GPU(a, b) ≈ Correlation_GPU(a, b)
        @test MSD_GPU([0.0, 1.0, 2.0, 3.0]) ≈ direct_msd([0.0, 1.0, 2.0, 3.0])
    end
end
