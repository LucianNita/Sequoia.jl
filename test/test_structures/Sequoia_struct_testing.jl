@testset "SEQUOIA_pb Struct Tests" begin   
   # Test Initialization of SEQUOIA_pb
    @testset "Initialization of SEQUOIA_pb" begin
        # Basic initialization with only the required parameter
        pb = SEQUOIA_pb(3)
        @test pb.nvar == 3
        @test pb.x0 == [0.0, 0.0, 0.0]
        @test pb.is_minimization == true
        @test pb.objective === nothing
        @test pb.gradient === nothing
        @test pb.constraints === nothing
        @test pb.jacobian === nothing
        @test pb.eqcon == Int[]
        @test pb.ineqcon == Int[]
        @test pb.solver_settings isa SEQUOIA_Settings
        @test pb.exitCode == :NotCalled
        @test pb.cutest_nlp === nothing

        # Initialization with custom fields
        obj = x -> sum(x.^2)
        gradient = x -> 2.0 * x
        constraints = x -> [x[1] + x[2] - 1.0]
        jacobian = x -> [1.0, 1.0]

        pb_custom = SEQUOIA_pb(
            2,
            x0 = [0.5, 0.5],
            is_minimization = false,
            objective = obj,
            gradient = gradient,
            constraints = constraints,
            jacobian = jacobian,
            eqcon = [1],
            ineqcon = Int[],
            solver_settings = SEQUOIA_Settings(:QPM, :Newton, false, 1e-6, 1000, 3600.0),
            exitCode = :OptimalityReached
        )

        @test pb_custom.nvar == 2
        @test pb_custom.x0 == [0.5, 0.5]
        @test pb_custom.is_minimization == false
        @test pb_custom.objective === obj
        @test pb_custom.gradient === gradient
        @test pb_custom.constraints === constraints
        @test pb_custom.jacobian === jacobian
        @test pb_custom.eqcon == [1]
        @test pb_custom.ineqcon == []
        @test pb_custom.solver_settings isa SEQUOIA_Settings
        @test pb_custom.exitCode == :OptimalityReached
    end

    # Test Initial Guess Setter
    @testset "Setting Initial Guess" begin
        pb = SEQUOIA_pb(3)

        # Valid input
        set_initial_guess!(pb, [1.0, 2.0, 3.0])
        @test pb.x0 == [1.0, 2.0, 3.0]

        # Invalid input (wrong dimension)
        @test_throws ArgumentError set_initial_guess!(pb, [1.0, 2.0])  # Should throw error
    end

    # Test Objective Function Setter
    @testset "Setting Objective Function" begin
        pb = SEQUOIA_pb(2)
        objective_fn = x -> (x[1] - 3.0)^2 + (x[2] - 2.0)^2

        # Valid objective function
        set_objective!(pb, objective_fn)
        @test pb.objective === objective_fn

        # Invalid objective function (output is not a scalar)
        @test_throws ArgumentError set_objective!(pb, x -> [x[1], x[2]])  # Should throw error
    end

    # Test Gradient Function and Automatic Differentiation
    @testset "Gradient Function Validation" begin
        pb = SEQUOIA_pb(2, x0 = [1.0, 2.0])
        objective_fn = x -> (x[1] - 3.0)^2 + (x[2] - 2.0)^2
        set_objective!(pb, objective_fn)

        # Ensure auto-differentiation if gradient not provided
        @test pb.gradient !== nothing
        @test pb.gradient([1.0, 2.0]) ≈ [-4.0, 0.0]

        # Custom gradient
        custom_gradient = x -> [2*(x[1]-3.0), 2*(x[2]-2.0)]
        set_objective!(pb, objective_fn, gradient = custom_gradient)
        @test pb.gradient === custom_gradient
    end

    # Test Constraints and Jacobian Validation
    @testset "Constraints and Jacobian Validation" begin
        pb = SEQUOIA_pb(2, x0 = [0.5, 0.5])

        # Valid constraints
        constraints_fn = x -> [x[1] + x[2] - 1.0]
        set_constraints!(pb, constraints_fn, [1], Int[])
        @test pb.constraints === constraints_fn

        # Error because the constraints function returns 2 constraints, but only 1 index is specified
        @test_throws ArgumentError set_constraints!(pb, x -> [x[1], x[2]], [1], Int[])

        # Valid Jacobian
        jacobian_fn = x -> [1.0 1.0]
        set_constraints!(pb, constraints_fn, [1], Int[], jacobian = jacobian_fn)
        @test pb.jacobian === jacobian_fn

        # Automatic differentiation for Jacobian
        pb.jacobian = nothing
        validate_constraints!(pb)  # Should set Jacobian using ForwardDiff
        @test pb.jacobian !== nothing
    end

    # Test Solver Settings
    @testset "Solver Settings" begin
        pb = SEQUOIA_pb(2)
        settings = SEQUOIA_Settings(:QPM, :Newton, false, 1e-6, 1000, 3600.0)

        # Valid settings
        set_solver_settings!(pb, settings)
        @test pb.solver_settings === settings

        # Invalid settings
        @test_throws ArgumentError set_solver_settings!(pb, "InvalidSetting")  # Should throw error
    end

    # Test Exit Code Validation
    @testset "Exit Code Validation" begin
        pb = SEQUOIA_pb(2)

        # Valid exit code
        update_exit_code!(pb, :OptimalityReached)
        @test pb.exitCode == :OptimalityReached

        # Invalid exit code
        @test_throws ArgumentError update_exit_code!(pb, :InvalidCode)  # Should throw error
    end

    # Test Reset Solution History
    @testset "Reset Solution History" begin
        # Initialize SEQUOIA_pb instance with nvar = 2
        pb = SEQUOIA_pb(2)

        # Creating a dummy SEQUOIA_Solution_step instance
        dummy_step = SEQUOIA_Solution_step(
            1,                              # Example outer iteration number
            0.01,                           # Dummy convergence metric
            :success,                       # Example solver status from the SolverStatus list
            0.5,                            # Example computation time in seconds
            2,                             # Example number of inner iterations
            [1.0, 2.0],                     # Example solution vector
            10.5,                           # Example objective function value
            [0.1, -0.2],                    # Example gradient vector
            [0.0],                          # Optional constraints vector
            [0.1, 0.5],                     # Optional solver parameters
            [[0.9, 1.8], [1.0, 2.0]]        # Optional history of x iterates
        )

        # Populate the solution history with the dummy step
        add_iterate!(pb.solution_history,dummy_step)

        # Verify that the solution history contains the dummy step
        @test length(pb.solution_history.iterates) == 1

        # Now, reset the solution history
        reset_solution_history!(pb)

        # Verify that the solution history is empty after the reset
        @test length(pb.solution_history.iterates) == 0
    end

    # Test Validation Functions
    @testset "Validation Functions" begin
        # Test validation of number of variables
        @test_throws ArgumentError SEQUOIA_pb(0)  # nvar should be positive

        pb = SEQUOIA_pb(2, x0 = [0.5, 0.5])

        # Test validation of initial guess
        @test_throws ArgumentError set_initial_guess!(pb, [1.0])  # Length mismatch
    end
end

@testset "SEQUOIA_pb examples tests" begin 
    # Test 1: Full Problem Setup with Constraints, Objective, and Initial Guess
    @testset "Full Problem Setup" begin
        pb = SEQUOIA_pb(
            2,
            x0 = [0.5, 0.5],
            is_minimization = true,
            solver_settings = SEQUOIA_Settings(:QPM, :LBFGS, false, 1e-6, 1000, 3600.0)
        )
    
        # Define the objective function
        objective_fn = x -> sum(x.^2)
        set_objective!(pb, objective_fn)
        
        # Check if the objective was set correctly
        @test pb.objective === objective_fn
    
        # Define constraints
        constraints_fn = x -> [x[1] + x[2] - 1.0]
        set_constraints!(pb, constraints_fn, [1], Int[])
    
        # Check if constraints were set correctly
        @test pb.constraints === constraints_fn
        @test pb.eqcon == [1]
        @test pb.ineqcon == Int[]
    
        # Check if initial guess was set correctly
        @test pb.x0 == [0.5, 0.5]
    end
    
    # Test 2: Unconstrained Minimization Problem with Automatic Differentiation
    @testset "Unconstrained Minimization Problem" begin
        pb = SEQUOIA_pb(2)
    
        # Define the objective function
        objective_fn = x -> (x[1] - 3.0)^2 + (x[2] - 2.0)^2
        set_objective!(pb, objective_fn)
    
        # Check if the objective was set correctly
        @test pb.objective === objective_fn
    
        # Check if the gradient was auto-computed (should not be `nothing`)
        @test pb.gradient !== nothing
    end
    
    # Test 3: Setting Solver Settings for a Feasibility Problem
    @testset "Feasibility Problem Setup" begin
        pb = SEQUOIA_pb(3)
    
        # Define solver settings
        settings = SEQUOIA_Settings(
            :QPM,
            :Newton,
            true,
            1e-6,
            500,
            1800.0
        )
        set_solver_settings!(pb, settings)
    
        # Check if the solver settings were set correctly
        @test pb.solver_settings === settings
    end
    
    # Test 4: Handling an Invalid Objective Function
    @testset "Invalid Objective Function Handling" begin
        pb = SEQUOIA_pb(2)
    
        # Attempt to set an invalid objective function
        @test_throws ArgumentError set_objective!(pb, x -> [x[1] + x[2]])
    end
    
    # Test 5: Setting Constraints and Jacobian with Automatic Differentiation
    @testset "Constraints and Jacobian with AutoDiff" begin
        pb = SEQUOIA_pb(2)
    
        # Define a simple constraint function
        constraints_fn = x -> [x[1]^2 + x[2] - 1.0]
        set_constraints!(pb, constraints_fn, [1], Int[])
    
        # Check if constraints were set correctly
        @test pb.constraints === constraints_fn
    
        # Check if Jacobian was auto-computed (should not be `nothing`)
        @test pb.jacobian !== nothing
    end
    
    # Test 6: Setting Custom Solver Settings
    @testset "Custom Solver Settings" begin
        pb = SEQUOIA_pb(3)
    
        # Define a simple objective function
        objective_fn = x -> sum(x.^2)
        set_objective!(pb, objective_fn)
    
        # Define custom solver settings
        custom_settings = SEQUOIA_Settings(:QPM, :Newton, false, 1e-7, 2000, 5000.0)
        set_solver_settings!(pb, custom_settings)
    
        # Check if the custom solver settings were applied
        @test pb.solver_settings === custom_settings
    end
    
    # Test 7: Resetting the Solution History
    @testset "Reset Solution History" begin
        pb = SEQUOIA_pb(2)
    
        # Define an objective function
        objective_fn = x -> sum(x.^2)
        set_objective!(pb, objective_fn)
    
        # Add a dummy step to the solution history
        dummy_step = SEQUOIA_Solution_step(
            1,
            0.1,
            :success,
            0.5,
            5,
            [0.0, 1.0],
            0.5,
            [0.1, -0.1]
        )
        add_iterate!(pb.solution_history, dummy_step)
    
        # Ensure that the history is not empty before reset
        @test length(pb.solution_history.iterates) > 0
    
        # Reset the solution history after solving
        reset_solution_history!(pb)
    
        # Check if history is empty after reset
        @test length(pb.solution_history.iterates) == 0
    end
    
    # Test 8: Handling Invalid Exit Codes
    @testset "Invalid Exit Codes" begin
        pb = SEQUOIA_pb(2)
    
        # Update the exit code with a valid value
        update_exit_code!(pb, :OptimalityReached)
    
        # Check if the exit code was set correctly
        @test pb.exitCode == :OptimalityReached
    
        # Now attempt to update the exit code with an invalid value
        @test_throws ArgumentError update_exit_code!(pb, :InvalidCode)
    end
    
end

