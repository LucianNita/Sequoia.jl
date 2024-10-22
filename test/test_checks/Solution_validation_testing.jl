@testset "SEQUOIA_Solution_step Validation Tests" begin
    # Valid case: Correct solution, should pass all validations
    valid_solution = SEQUOIA_Solution_step(
        10,                     # outer_iteration_number
        1e-6,                   # convergence_metric
        :success,               # solver_status
        0.02,                   # inner_comp_time
        5,                      # num_inner_iterations
        [1.0, 2.0],             # x
        0.5,                    # fval
        [0.1, 0.2],             # gval
        nothing,                # cval
        [0.5, 0.3],             # solver_params
        [[1.0, 2.0], [1.1, 2.1], [1.2, 2.2], [1.3, 2.3], [1.4, 2.4]] # x_iterates
    )
    @test validate_sequoia_solution!(valid_solution) == nothing  # Should not throw any error

    # Invalid case: Empty solution vector `x`
    @test_throws ArgumentError SEQUOIA_Solution_step(
        10, 1e-6, :success, 0.02, 5, Float64[], 0.5, [0.1, 0.2]
    )  # Should fail due to empty x

    # Invalid case: Empty gradient vector `gval`
    @test_throws ArgumentError SEQUOIA_Solution_step(
        10, 1e-6, :success, 0.02, 5, [1.0, 2.0], 0.5, Float64[]
    )  # Should fail due to empty gval

    # Invalid case: Non-matching number of elements in `x` and `gval`
    @test_throws ArgumentError SEQUOIA_Solution_step(
        10, 1e-6, :success, 0.02, 5, [1.0, 2.0], 0.5, [0.1]
    )  # Should fail due to mismatched length of x and gval

    # Invalid case: Non-numeric `fval`
    @test_throws MethodError SEQUOIA_Solution_step(
        10, 1e-6, :success, 0.02, 5, [1.0, 2.0], "invalid", [0.1, 0.2]
    )  # Should fail due to non-numeric fval

    # Invalid case: Non-numeric `convergence_metric`
    @test_throws MethodError SEQUOIA_Solution_step(
        10, "invalid", :success, 0.02, 5, [1.0, 2.0], 0.5, [0.1, 0.2]
    )  # Should fail due to non-numeric convergence_metric

    # Invalid case: Negative `inner_comp_time`
    @test_throws ArgumentError SEQUOIA_Solution_step(
        10, 1e-6, :success, -0.01, 5, [1.0, 2.0], 0.5, [0.1, 0.2]
    )  # Should fail due to negative inner_comp_time

    # Invalid case: Negative `outer_iteration_number`
    @test_throws ArgumentError SEQUOIA_Solution_step(
        -1, 1e-6, :success, 0.02, 5, [1.0, 2.0], 0.5, [0.1, 0.2]
    )  # Should fail due to negative outer_iteration_number

    # Invalid case: Negative `num_inner_iterations`
    @test_throws ArgumentError SEQUOIA_Solution_step(
        10, 1e-6, :success, 0.02, -5, [1.0, 2.0], 0.5, [0.1, 0.2]
    )  # Should fail due to negative num_inner_iterations

    # Invalid case: Non-numeric values in `cval`
    @test_throws MethodError SEQUOIA_Solution_step(
        10, 1e-6, :success, 0.02, 5, [1.0, 2.0], 0.5, [0.1, 0.2], [0.1, "invalid"]
    )  # Should fail due to invalid non-numeric cval

    # Invalid case: Invalid solver status
    @test_throws ArgumentError SEQUOIA_Solution_step(
        10, 1e-6, :invalid_status, 0.02, 5, [1.0, 2.0], 0.5, [0.1, 0.2]
    )  # Should fail due to invalid solver status

    # Invalid case: Incorrect number of `x_iterates`
    @test_throws ArgumentError SEQUOIA_Solution_step(
        10, 1e-6, :success, 0.02, 5, [1.0, 2.0], 0.5, [0.1, 0.2], nothing, nothing, [[1.0, 2.0]]  # Only 1 x_iterate instead of 5
    )  # Should fail due to mismatched x_iterates length

    # Invalid case: Mismatched dimensions in `x_iterates`
    @test_throws ArgumentError SEQUOIA_Solution_step(
        10, 1e-6, :success, 0.02, 5, [1.0, 2.0], 0.5, [0.1, 0.2], nothing, nothing, [[1.0], [1.1, 2.1], [1.2, 2.2], [1.3, 2.3], [1.4, 2.4]]
    )  # Should fail due to mismatched dimensions in x_iterates

    # Invalid case: Solution vector `x` of type any
    @test_throws MethodError SEQUOIA_Solution_step(
        10, 1e-6, :success, 0.02, 5, [], 0.5, [0.1, 0.2]
    )  # Should fail due to empty x
end
