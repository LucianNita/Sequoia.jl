# Test the SEQUOIA_Solution_step constructors and validation
@testset "SEQUOIA_Solution_step Constructors and Validation" begin

    # Unit tests for SEQUOIA_Solution_step
    @testset "SEQUOIA_Solution_step Constructor - Valid Inputs" begin
        # Full constructor test with all fields
        step_full = SEQUOIA_Solution_step(
            10,                    # outer_iteration_number
            1e-6,                  # convergence_metric
            :success,              # solver_status
            0.02,                  # inner_comp_time
            5,                     # num_inner_iterations
            [1.0, 2.0],            # x
            0.5,                   # fval
            [0.1, 0.2],            # gval
            nothing,               # cval
            [0.5, 0.3],            # solver_params
            [[1.0, 2.0], [1.1, 2.1], [1.2, 2.2], [1.3, 2.3], [1.4, 2.4]] # x_iterates matching num_inner_iterations
        )

        @test step_full.outer_iteration_number == 10
        @test step_full.convergence_metric == 1e-6
        @test step_full.solver_status == :success
        @test step_full.inner_comp_time == 0.02
        @test step_full.num_inner_iterations == 5
        @test step_full.x == [1.0, 2.0]
        @test step_full.fval == 0.5
        @test step_full.gval == [0.1, 0.2]
        @test step_full.cval == nothing
        @test step_full.solver_params == [0.5, 0.3]
        @test step_full.x_iterates == [[1.0, 2.0], [1.1, 2.1], [1.2, 2.2], [1.3, 2.3], [1.4, 2.4]]

        # Intermediate constructor test
        step_intermediate = SEQUOIA_Solution_step(
            10,                    # outer_iteration_number
            1e-6,                  # convergence_metric
            :failed,               # solver_status
            0.03,                  # inner_comp_time
            6,                     # num_inner_iterations
            [1.1, 2.1],            # x
            0.45,                  # fval
            [0.05, 0.15],          # gval
            [0.1, 0.2],            # cval
            [0.01, 0.02]           # solver_params
        )

        @test step_intermediate.outer_iteration_number == 10
        @test step_intermediate.convergence_metric == 1e-6
        @test step_intermediate.solver_status == :failed
        @test step_intermediate.inner_comp_time == 0.03
        @test step_intermediate.num_inner_iterations == 6
        @test step_intermediate.x == [1.1, 2.1]
        @test step_intermediate.fval == 0.45
        @test step_intermediate.gval == [0.05, 0.15]
        @test step_intermediate.cval == [0.1, 0.2]
        @test step_intermediate.solver_params == [0.01, 0.02]
        @test step_intermediate.x_iterates == nothing  # Default to nothing for x_iterates

        # Minimal constructor test with only mandatory fields
        step_minimal = SEQUOIA_Solution_step(
            0,                     # outer_iteration_number
            0.0,                   # convergence_metric
            :success,              # solver_status
            0.0,                   # inner_comp_time
            0,                     # num_inner_iterations
            [1.0, 2.0],            # x
            0.5,                   # fval
            [0.1, 0.2]             # gval
        )

        @test step_minimal.outer_iteration_number == 0
        @test step_minimal.convergence_metric == 0.0
        @test step_minimal.solver_status == :success
        @test step_minimal.inner_comp_time == 0.0
        @test step_minimal.num_inner_iterations == 0
        @test step_minimal.x == [1.0, 2.0]
        @test step_minimal.fval == 0.5
        @test step_minimal.gval == [0.1, 0.2]
        @test step_minimal.cval == nothing
        @test step_minimal.solver_params == nothing
        @test step_minimal.x_iterates == nothing
    end

    @testset "SEQUOIA_Solution_step Constructor - Invalid Inputs" begin
        # Invalid cases
        @test_throws ArgumentError SEQUOIA_Solution_step(10, 1e-6, :success, 0.02, 5, Float64[], 0.5, [0.1, 0.2]) # Empty solution vector
        @test_throws ArgumentError SEQUOIA_Solution_step(10, 1e-6, :success, 0.02, 5, [1.0], 0.5, Float64[])       # Empty gradient vector
        @test_throws MethodError SEQUOIA_Solution_step(10, 1e-6, :success, 0.02, 5, [1.0], "0.5", [0.1])           # Invalid non-numeric fval

        # Invalid constraint value
        @test_throws MethodError SEQUOIA_Solution_step(10, 1e-6, :success, 0.02, 5, [1.0], 0.5, [0.1], [0.1, "invalid"])

        # Invalid x_iterates size
        @test_throws ArgumentError SEQUOIA_Solution_step(10, 1e-6, :success, 0.02, 5, [1.0, 2.0], 0.5, [0.1, 0.2], nothing, nothing, [[1.0, 2.0]])

        # Invalid solver status
        @test_throws ArgumentError SEQUOIA_Solution_step(10, 1e-6, :invalid_status, 0.02, 5, [1.0, 2.0], 0.5, [0.1, 0.2])
    end

    # Example 1: Full Constructor
    @testset "Example 1: Full Constructor" begin
        solution_full = SEQUOIA_Solution_step(
            10,                       # outer_iteration_number
            1e-6,                     # convergence_metric
            :success,                 # solver_status
            0.02,                     # inner_comp_time
            5,                        # num_inner_iterations
            [1.0, 2.0],               # x
            0.5,                      # fval
            [0.1, 0.2],               # gval
            nothing,                  # cval
            [0.5, 0.3],               # solver_params
            [[1.0, 2.0], [1.1, 2.1], [1.2, 2.2], [1.3, 2.3], [1.4, 2.4]]  # x_iterates
        )

        @test solution_full.outer_iteration_number == 10
        @test solution_full.convergence_metric == 1e-6
        @test solution_full.solver_status == :success
        @test solution_full.inner_comp_time == 0.02
        @test solution_full.num_inner_iterations == 5
        @test solution_full.x == [1.0, 2.0]
        @test solution_full.fval == 0.5
        @test solution_full.gval == [0.1, 0.2]
        @test solution_full.cval == nothing
        @test solution_full.solver_params == [0.5, 0.3]
        @test solution_full.x_iterates == [[1.0, 2.0], [1.1, 2.1], [1.2, 2.2], [1.3, 2.3], [1.4, 2.4]]
    end

    # Example 2: Intermediate Constructor
    @testset "Example 2: Intermediate Constructor" begin
        solution_intermediate = SEQUOIA_Solution_step(
            10,                       # outer_iteration_number
            1e-6,                     # convergence_metric
            :failed,                  # solver_status
            0.03,                     # inner_comp_time
            6,                        # num_inner_iterations
            [1.1, 2.1],               # x
            0.45,                     # fval
            [0.05, 0.15],             # gval
            [0.1, 0.2],               # cval
            [0.01, 0.02]              # solver_params
        )

        @test solution_intermediate.outer_iteration_number == 10
        @test solution_intermediate.convergence_metric == 1e-6
        @test solution_intermediate.solver_status == :failed
        @test solution_intermediate.inner_comp_time == 0.03
        @test solution_intermediate.num_inner_iterations == 6
        @test solution_intermediate.x == [1.1, 2.1]
        @test solution_intermediate.fval == 0.45
        @test solution_intermediate.gval == [0.05, 0.15]
        @test solution_intermediate.cval == [0.1, 0.2]
        @test solution_intermediate.solver_params == [0.01, 0.02]
        @test solution_intermediate.x_iterates == nothing
    end

    # Example 3: Minimal Constructor
    @testset "Example 3: Minimal Constructor" begin
        solution_minimal = SEQUOIA_Solution_step(
            0,                        # outer_iteration_number
            0.0,                      # convergence_metric
            :success,                 # solver_status
            0.0,                      # inner_comp_time
            0,                        # num_inner_iterations
            [1.0, 2.0],               # x
            0.5,                      # fval
            [0.1, 0.2]                # gval
        )

        @test solution_minimal.outer_iteration_number == 0
        @test solution_minimal.convergence_metric == 0.0
        @test solution_minimal.solver_status == :success
        @test solution_minimal.inner_comp_time == 0.0
        @test solution_minimal.num_inner_iterations == 0
        @test solution_minimal.x == [1.0, 2.0]
        @test solution_minimal.fval == 0.5
        @test solution_minimal.gval == [0.1, 0.2]
        @test solution_minimal.cval == nothing
        @test solution_minimal.solver_params == nothing
        @test solution_minimal.x_iterates == nothing
    end

    # Example 4: Error Handling with Invalid Inputs
    @testset "Example 4: Error Handling with Invalid Inputs" begin
        @test_throws ArgumentError SEQUOIA_Solution_step(
            10,                       # outer_iteration_number
            1e-6,                     # convergence_metric
            :success,                 # solver_status
            0.02,                     # inner_comp_time
            5,                        # num_inner_iterations
            Float64[],                # Invalid: empty solution vector `x`
            0.5,                      # fval
            [0.1, 0.2],               # gval
            nothing,                  # cval
            [0.5, 0.3],               # solver_params
            [[1.0, 2.0]]              # x_iterates
        )
    end

    # Example 5: Handling Mismatched Dimensions in x_iterates
    @testset "Example 5: Mismatched Dimensions in x_iterates" begin
        @test_throws ArgumentError SEQUOIA_Solution_step(
            10,                       # outer_iteration_number
            1e-6,                     # convergence_metric
            :success,                 # solver_status
            0.02,                     # inner_comp_time
            5,                        # num_inner_iterations
            [1.0, 2.0],               # x
            0.5,                      # fval
            [0.1, 0.2],               # gval
            nothing,                  # cval
            [0.5, 0.3],               # solver_params
            [[1.0, 2.0], [1.1, 2.1]]  # Invalid: fewer x_iterates than num_inner_iterations
        )
    end
end