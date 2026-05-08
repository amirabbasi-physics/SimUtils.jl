@testset "Correlation" begin
    a = [1.0, -2.0, 3.0, 0.5]
    b = [2.0, 1.5, -1.0, 4.0]

    @test Correlation(a) ≈ direct_correlation(a)
    @test Correlation(a, b) ≈ direct_correlation(a, b)
    @test correlation(a, b) ≈ Correlation(a, b)
    @test Correlation(a, b; subtract_mean=true) ≈ direct_correlation(a, b; subtract_mean=true)
    @test correlation(a; subtract_mean=true) ≈ Correlation(a; subtract_mean=true)

    int_result = try
        Correlation([1, 2, 3, 4]; subtract_mean=true)
    catch err
        err
    end
    @test !(int_result isa Exception)
    if !(int_result isa Exception)
        @test int_result ≈ direct_correlation([1, 2, 3, 4]; subtract_mean=true)
    end
end
