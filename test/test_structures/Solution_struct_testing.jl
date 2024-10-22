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

#=
# Test SEQUOIA_Iterates and the add_step! and getter functions
@testset "SEQUOIA_Iterates, add_step!, and getter functions" begin
    # Create empty iterates
    iterates = SEQUOIA_Iterates()
    @test isempty(iterates.steps)

    # Add a step
    step1 = SEQUOIA_Solution_step([1.0, 2.0], 0.5, [0.1, 0.2], nothing, 0.01, 1e-6, 10, 5, 0.02, :success, [[1.0, 2.0], [1.1, 2.1]])
    add_step!(iterates, step1)
    @test length(iterates.steps) == 1
    @test iterates.steps[1] == step1

    # Add another step
    step2 = SEQUOIA_Solution_step([1.1, 2.1], 0.45, [0.05, 0.15], [0.1, 0.2], 0.02, 1e-7, 11, 6, 0.03, :failed, [[1.1, 2.1], [1.2, 2.2]])
    add_step!(iterates, step2)
    @test length(iterates.steps) == 2
    @test iterates.steps[2] == step2

    # Test get_all_solutions
    all_solutions = get_all_solutions(iterates)
    @test all_solutions == [[1.0, 2.0], [1.1, 2.1]]

    # Test get_all_fvals
    all_fvals = get_all_fvals(iterates)
    @test all_fvals == [0.5, 0.45]

    # Test get_all_convergence_metrics
    all_convergence_metrics = get_all_convergence_metrics(iterates)
    @test all_convergence_metrics == [1e-6, 1e-7]

    # Test get_all_cvals
    all_cvals = get_all_cvals(iterates)
    @test all_cvals == [nothing, [0.1, 0.2]]

    # Test get_all_step_sizes
    all_step_sizes = get_all_step_sizes(iterates)
    @test all_step_sizes == [0.01, 0.02]

    # Test get_all_outer_iteration_numbers
    all_outer_iteration_numbers = get_all_outer_iteration_numbers(iterates)
    @test all_outer_iteration_numbers == [10, 11]

    # Test get_all_inner_iterations
    all_inner_iterations = get_all_inner_iterations(iterates)
    @test all_inner_iterations == [5, 6]

    # Test get_all_inner_comp_times
    all_inner_comp_times = get_all_inner_comp_times(iterates)
    @test all_inner_comp_times == [0.02, 0.03]

    # Test get_all_solver_statuses
    all_solver_statuses = get_all_solver_statuses(iterates)
    @test all_solver_statuses == [:success, :failed]

    # Test new functionality: x_iterates history
    @test step1.x_iterates == [[1.0, 2.0], [1.1, 2.1]]
    @test step2.x_iterates == [[1.1, 2.1], [1.2, 2.2]]

    # Edge case: empty SEQUOIA_Iterates
    empty_iterates = SEQUOIA_Iterates()
    @test get_all_solutions(empty_iterates) == []
    @test get_all_fvals(empty_iterates) == []
    @test get_all_convergence_metrics(empty_iterates) == []
    @test get_all_cvals(empty_iterates) == []
    @test get_all_step_sizes(empty_iterates) == []
    @test get_all_outer_iteration_numbers(empty_iterates) == []
    @test get_all_inner_iterations(empty_iterates) == []
    @test get_all_inner_comp_times(empty_iterates) == []
    @test get_all_solver_statuses(empty_iterates) == []
end
=#