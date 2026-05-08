@testset "Fourier Transforms" begin
    t = 0.2 .* collect(0:7)
    x = cos.(2π .* 1.25 .* t) .+ 0.1 .* sin.(2π .* 0.75 .* t)

    ω_expected, X_expected = expected_ft(t, x)
    ω, X = FT(t, x)

    @test ω ≈ ω_expected
    @test X ≈ X_expected atol=1e-10 rtol=1e-10
    @test FT(t, x, 1e-8, false) ≈ X

    t0 = -0.3
    t_shifted = t0 .+ 0.1 .* collect(0:7)
    x_shifted = [0.4, -1.2, 0.7, 0.0, 1.1, -0.6, 0.2, 0.9]
    ω_ref, X_ref = expected_ft(t_shifted, x_shifted)

    inverse_result = try
        iFT(ω_ref, X_ref; t0=t_shifted[1])
    catch err
        err
    end
    @test !(inverse_result isa Exception)
    if !(inverse_result isa Exception)
        t_rec, x_rec = inverse_result
        @test t_rec ≈ t_shifted atol=1e-12 rtol=1e-12
        @test real.(x_rec) ≈ x_shifted atol=1e-10 rtol=1e-10
        @test maximum(abs.(imag.(x_rec))) ≤ 1e-10
    end

    @test_throws ErrorException FT([0.0, 0.3, 0.7], [1.0, 2.0, 3.0])
end
