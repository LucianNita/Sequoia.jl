@testset "SEQUOIA_Solution_step Validation Tests" begin

    # Valid case: Correct solution, should pass all validations
    @testset "Valid Solution Step" begin
        valid_solution = SEQUOIA_Solution_step(
            10,                     # outer_iteration_number
            1e-6,                   # convergence_metric
            :first_order,           # solver_status
            0.02,                   # inner_comp_time
            4,                      # num_inner_iterations
            [1.0, 2.0],             # x
            0.5,                    # fval
            [0.1, 0.2],             # gval
            nothing,                # cval
            [0.5, 0.3],             # solver_params
            [[1.0, 2.0], [1.1, 2.1], [1.2, 2.2], [1.3, 2.3], [1.4, 2.4]] # x_iterates
        )
        @test validate_sequoia_solution!(valid_solution) === nothing  # Should not throw any error
    end

    # Invalid case: Empty solution vector `x`
    @testset "Invalid Empty Solution Vector x" begin
        @test_throws ArgumentError validate_sequoia_solution!(SEQUOIA_Solution_step(
            10, 1e-6, :first_order, 0.02, 5, Float64[], 0.5, [0.1, 0.2]
        ))  # Should fail due to empty x
    end

    # Invalid case: Negative convergence metric `convergence_metric`
    @testset "Negative convergence metric" begin
        @test_throws ArgumentError validate_sequoia_solution!(SEQUOIA_Solution_step(
            10, -1e-6, :first_order, 0.02, 5, [1.0, 2.0], 0.5, [0.1, 0.2]
        ))  # Should fail due to negative convergence metric
    end

    # Invalid case: Empty gradient vector `gval`
    @testset "Invalid Empty Gradient Vector gval" begin
        @test_throws ArgumentError validate_sequoia_solution!(SEQUOIA_Solution_step(
            10, 1e-6, :first_order, 0.02, 5, [1.0, 2.0], 0.5, Float64[]
        ))  # Should fail due to empty gval
    end

    # Invalid case: Non-matching number of elements in `x` and `gval`
    @testset "Mismatched x and gval Lengths" begin
        @test_throws ArgumentError validate_sequoia_solution!(SEQUOIA_Solution_step(
            10, 1e-6, :first_order, 0.02, 5, [1.0, 2.0], 0.5, [0.1]
        ))  # Should fail due to mismatched length of x and gval
    end

    # Invalid case: Negative `inner_comp_time`
    @testset "Negative Inner Computation Time" begin
        @test_throws ArgumentError validate_sequoia_solution!(SEQUOIA_Solution_step(
            10, 1e-6, :first_order, -0.01, 5, [1.0, 2.0], 0.5, [0.1, 0.2]
        ))  # Should fail due to negative inner_comp_time
    end

    # Invalid case: Negative `outer_iteration_number`
    @testset "Negative Outer Iteration Number" begin
        @test_throws ArgumentError validate_sequoia_solution!(SEQUOIA_Solution_step(
            -1, 1e-6, :first_order, 0.02, 5, [1.0, 2.0], 0.5, [0.1, 0.2]
        ))  # Should fail due to negative outer_iteration_number
    end

    # Invalid case: Negative `num_inner_iterations`
    @testset "Negative inner number of iterations" begin
        @test_throws ArgumentError validate_sequoia_solution!(SEQUOIA_Solution_step(
            10, 1e-6, :first_order, 0.02, -5, [1.0, 2.0], 0.5, [0.1, 0.2]
        ))  # Should fail due to negative inner number of iterations
    end

    # Invalid case: Non-numeric `convergence_metric`
    @testset "Invalid Convergence Metric Type" begin
        @test_throws MethodError validate_sequoia_solution!(SEQUOIA_Solution_step(
            10, "invalid", :first_order, 0.02, 5, [1.0, 2.0], 0.5, [0.1, 0.2]
        ))  # Should fail due to non-numeric convergence_metric
    end

    # Invalid case: Non-numeric values in `cval`
    @testset "Invalid Constraint Values cval" begin
        @test_throws MethodError validate_sequoia_solution!(SEQUOIA_Solution_step(
            10, 1e-6, :first_order, 0.02, 5, [1.0, 2.0], 0.5, [0.1, 0.2], [0.1, "invalid"]
        ))  # Should fail due to invalid non-numeric cval
    end

    # Invalid case: Incorrect number of `x_iterates`
    @testset "Incorrect Number of x_iterates" begin
        @test_throws ArgumentError validate_sequoia_solution!(SEQUOIA_Solution_step(
            10, 1e-6, :first_order, 0.02, 5, [1.0, 2.0], 0.5, [0.1, 0.2], nothing, nothing,
            [[1.0, 2.0]]  # Only 1 x_iterate instead of 5
        ))  # Should fail due to mismatched x_iterates length
    end

    # Invalid case: Mismatched dimensions in `x_iterates`
    @testset "Mismatched Dimensions in x_iterates" begin
        @test_throws ArgumentError validate_sequoia_solution!(SEQUOIA_Solution_step(
            10, 1e-6, :first_order, 0.02, 5, [1.0, 2.0], 0.5, [0.1, 0.2], nothing, nothing,
            [[1.0,1.0], [1.1, 2.1], [1.2, 2.2], [1.3, 2.3], [1.4, 2.4]]  # Invalid dimensions in x_iterates
        ))
    end

    # Invalid case: `x_iterates` should contain vectors of the same size as `x`
    @testset "Wrong Dimensions in x_iterates components" begin
        @test_throws ArgumentError validate_sequoia_solution!(SEQUOIA_Solution_step(
            10, 1e-6, :first_order, 0.02, 4, [1.0, 2.0], 0.5, [0.1, 0.2], nothing, nothing,
            [[1.0, 1.0], [1.1, 2.1], [1.2, 2.2], [1.3, 2.3], [1.4]]  # Invalid dimensions in x_iterates
        ))
    end

    # Invalid case: Invalid solver status
    @testset "Invalid Solver Status" begin
        @test_throws ArgumentError validate_sequoia_solution!(SEQUOIA_Solution_step(
            10, 1e-6, :invalid_status, 0.02, 5, [1.0, 2.0], 0.5, [0.1, 0.2]
        ))  # Should fail due to invalid solver status
    end
end
