@testset "MSD" begin
    @test MSD(fill(3.0, 5)) ≈ zeros(5) atol=1e-12

    oscillatory = [1.0, 2.0, 1.0, 2.0]
    @test MSD(oscillatory) ≈ direct_msd(oscillatory)

    drift = [0.0, 1.0, 2.0, 3.0]
    @test MSD(drift) ≈ direct_msd(drift)

    curved = [0.0, 1.0, 4.0, 9.0]
    @test MSD(curved) ≈ direct_msd(curved)
end
