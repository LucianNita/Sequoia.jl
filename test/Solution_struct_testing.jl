# Test the SEQUOIA_Solution_step constructors and validation
@testset "SEQUOIA_Solution_step constructors and validation" begin
    # Basic constructor test with all fields
    step = SEQUOIA_Solution_step([1.0, 2.0], 0.5, [0.1, 0.2], nothing, 0.01, 1e-6, 10, 5, 0.02, :success, [[1.0, 2.0], [1.1, 2.1]])
    @test step.x == [1.0, 2.0]
    @test step.fval == 0.5
    @test step.gval == [0.1, 0.2]
    @test step.cval == nothing
    @test step.step_size == 0.01
    @test step.convergence_metric == 1e-6
    @test step.outer_iteration_number == 10
    @test step.num_inner_iterations == 5
    @test step.inner_comp_time == 0.02
    @test step.solver_status == :success
    @test step.x_iterates == [[1.0, 2.0], [1.1, 2.1]]  # New check for x_iterates

    # Constructor without optional fields
    step_minimal = SEQUOIA_Solution_step([1.0, 2.0], 0.5, [0.1, 0.2])
    @test step_minimal.x == [1.0, 2.0]
    @test step_minimal.fval == 0.5
    @test step_minimal.gval == [0.1, 0.2]
    @test step_minimal.cval == nothing
    @test step_minimal.step_size == nothing
    @test step_minimal.convergence_metric == 0.0
    @test step_minimal.outer_iteration_number == 0
    @test step_minimal.num_inner_iterations == nothing
    @test step_minimal.inner_comp_time == 0.0
    @test step_minimal.solver_status == :success
    @test step_minimal.x_iterates == [[1.0, 2.0]]  # New check for x_iterates, starts with x

    # Invalid cases
    @test_throws AssertionError SEQUOIA_Solution_step([], 0.5, [0.1, 0.2]) # Empty solution vector
    @test_throws AssertionError SEQUOIA_Solution_step([1.0], 0.5, [])      # Empty gradient vector
    @test_throws AssertionError SEQUOIA_Solution_step([1.0], "0.5", [0.1]) # Invalid non-numeric fval

    # Invalid constraint value
    @test_throws AssertionError SEQUOIA_Solution_step([1.0], 0.5, [0.1], [0.1, "invalid"])
    
    # Invalid step size (negative)
    @test_throws AssertionError SEQUOIA_Solution_step([1.0], 0.5, [0.1], nothing, -0.01)

    # Invalid solver status
    @test_throws AssertionError SEQUOIA_Solution_step([1.0], 0.5, [0.1], nothing, 0.01, 1e-6, 10, 5, 0.02, :invalid_status)
end

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
