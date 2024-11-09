@testset "SEQUOIA_Solution_step Unit Tests" begin

    # Test 1: A Successful Iteration
    @testset "Successful Iteration" begin
        solution = SEQUOIA_Solution_step(
            5,                      # Outer iteration number
            1e-8,                   # Convergence metric
            :first_order,           # Solver status
            0.02,                   # Inner computation time
            2,                      # Number of inner iterations
            [1.0, 2.0],             # Solution vector
            -5.0,                   # Objective value
            [0.0, 0.0],             # Gradient vector
            nothing,                # No constraints
            [0.5, 0.3],             # Solver parameters
            [[0.8, 1.5], [0.9, 1.8], [1.0, 2.0]] # Inner x iterates history
        )
        @test solution.outer_iteration_number == 5
        @test solution.convergence_metric == 1e-8
        @test solution.solver_status == :first_order
        @test solution.inner_comp_time == 0.02
        @test solution.num_inner_iterations == 2
        @test solution.x == [1.0, 2.0]
        @test solution.fval == -5.0
        @test solution.gval == [0.0, 0.0]
        @test solution.cval == nothing
        @test solution.solver_params == [0.5, 0.3]
        @test solution.x_iterates == [[0.8, 1.5], [0.9, 1.8], [1.0, 2.0]]
    end

    # Test 2: Iteration Terminated Due to Small Step Size
    @testset "Small Step Termination" begin
        solution = SEQUOIA_Solution_step(
            8,                      # Outer iteration number
            1e-4,                   # Convergence metric
            :small_step,            # Solver status
            0.05,                   # Inner computation time
            12,                     # Number of inner iterations
            [0.99, 1.01],           # Solution vector
            -4.8,                   # Objective value
            [0.01, -0.01],          # Gradient vector
            [0.02],                 # Constraints
            [0.8, 0.5],             # Solver parameters
            nothing                 # No x iterates history
        )
        @test solution.outer_iteration_number == 8
        @test solution.convergence_metric == 1e-4
        @test solution.solver_status == :small_step
        @test solution.inner_comp_time == 0.05
        @test solution.num_inner_iterations == 12
        @test solution.x == [0.99, 1.01]
        @test solution.fval == -4.8
        @test solution.gval == [0.01, -0.01]
        @test solution.cval == [0.02]
        @test solution.solver_params == [0.8, 0.5]
        @test solution.x_iterates == nothing
    end

    # Test 3: Handling an Unbounded Problem
    @testset "Unbounded Problem" begin
        solution = SEQUOIA_Solution_step(
            15,                     # Outer iteration number
            Inf,                    # Convergence metric indicating divergence
            :unbounded,             # Solver status
            0.1,                    # Inner computation time
            20,                     # Number of inner iterations
            [100.0, -100.0],        # Solution vector trending towards infinity
            -Inf,                   # Objective value diverging
            [10.0, -10.0],          # Gradient vector
            nothing,                # No constraints
            [1.0, 1.5],             # Solver parameters
            nothing                 # No x iterates history
        )
        @test solution.outer_iteration_number == 15
        @test solution.convergence_metric == Inf
        @test solution.solver_status == :unbounded
        @test solution.inner_comp_time == 0.1
        @test solution.num_inner_iterations == 20
        @test solution.x == [100.0, -100.0]
        @test solution.fval == -Inf
        @test solution.gval == [10.0, -10.0]
        @test solution.cval == nothing
        @test solution.solver_params == [1.0, 1.5]
        @test solution.x_iterates == nothing
    end

    # Test 4: Handling an Infeasible Problem
    @testset "Infeasible Problem" begin
        solution = SEQUOIA_Solution_step(
            20,                     # Outer iteration number
            1.0,                    # Convergence metric
            :infeasible,            # Solver status
            0.2,                    # Inner computation time
            15,                     # Number of inner iterations
            [0.5, 0.5],             # Solution vector
            10.0,                   # Objective value
            [2.0, 2.0],             # Gradient vector
            [1.0, -1.5],            # Constraints
            nothing,                # No solver parameters
            nothing                 # No x iterates history
        )
        @test solution.outer_iteration_number == 20
        @test solution.convergence_metric == 1.0
        @test solution.solver_status == :infeasible
        @test solution.inner_comp_time == 0.2
        @test solution.num_inner_iterations == 15
        @test solution.x == [0.5, 0.5]
        @test solution.fval == 10.0
        @test solution.gval == [2.0, 2.0]
        @test solution.cval == [1.0, -1.5]
        @test solution.solver_params == nothing
        @test solution.x_iterates == nothing
    end

    # Test 5: Debugging a Solver Step
    @testset "Debugging Solver Step" begin
        solution = SEQUOIA_Solution_step(
            3,                      # Outer iteration number
            1e-3,                   # Convergence metric
            :acceptable,            # Solver status
            0.15,                   # Inner computation time
            2,                      # Number of inner iterations
            [1.2, 1.8],             # Solution vector
            -3.5,                   # Objective value
            [0.02, -0.03],          # Gradient vector
            [0.0, 0.01],            # Constraints
            [0.7, 0.4],             # Solver parameters
            [[1.1, 1.7], [1.15, 1.75], [1.2, 1.8]] # x iterates history
        )
        @test solution.outer_iteration_number == 3
        @test solution.convergence_metric == 1e-3
        @test solution.solver_status == :acceptable
        @test solution.inner_comp_time == 0.15
        @test solution.num_inner_iterations == 2
        @test solution.x == [1.2, 1.8]
        @test solution.fval == -3.5
        @test solution.gval == [0.02, -0.03]
        @test solution.cval == [0.0, 0.01]
        @test solution.solver_params == [0.7, 0.4]
        @test solution.x_iterates == [[1.1, 1.7], [1.15, 1.75], [1.2, 1.8]]
    end

    # Test 6: Minimal Example with Default Fields
    @testset "Minimal Example" begin
        solution = SEQUOIA_Solution_step(
            1,                      # Outer iteration number
            0.01,                   # Convergence metric
            :max_iter,              # Solver status
            0.05,                   # Inner computation time
            5,                      # Number of inner iterations
            [0.0, 1.0],             # Solution vector
            0.2,                    # Objective value
            [0.1, -0.1]             # Gradient vector
        )
        @test solution.outer_iteration_number == 1
        @test solution.convergence_metric == 0.01
        @test solution.solver_status == :max_iter
        @test solution.inner_comp_time == 0.05
        @test solution.num_inner_iterations == 5
        @test solution.x == [0.0, 1.0]
        @test solution.fval == 0.2
        @test solution.gval == [0.1, -0.1]
        @test solution.cval == nothing
        @test solution.solver_params == nothing
        @test solution.x_iterates == nothing
    end

end
