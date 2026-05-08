@testset "Padding And Tapers" begin
    @test pad_to_power_of_two([1.0, 2.0, 3.0], 1.0) == [0.0, 1.0, 2.0, 0.0]
    @test pad_to_power_of_two([1.0, 2.0, 3.0, 4.0], 0.0) == [1.0, 2.0, 3.0, 4.0]

    data1 = ones(5)
    @test apply_taper!(data1, 2, 4) === nothing
    @test data1 ≈ [1.0, 0.0, 0.5, 1.0, 1.0]

    data2 = ones(5)
    @test apply_taper2!(data2, 2, 4) === nothing
    @test data2 ≈ [1.0, 1.0, cos(pi / 4), 0.0, 1.0]
end
