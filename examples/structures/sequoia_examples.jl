"""
# SEQUOIA_pb Examples

This file contains example use cases for the `SEQUOIA_pb` struct and related functions, demonstrating its usage in different scenarios:

1. Minimal initialization with just the number of variables (`nvar`) specified.
2. Basic initialization of an optimization problem with an objective function.
3. Adding a provided gradient or using automatic differentiation for the gradient.
4. Adding constraints and their Jacobian or using automatic differentiation for missing Jacobians.
5. Automatic differentiation for a missing gradient.
6. Automatic differentiation for a missing Jacobian.
7. Modifying solver settings for the optimization problem.
8. Resetting the solution history of a problem.
9. Using `set_initial_guess!` to update the initial guess.
10. Using `set_objective!` to update the objective function and gradient.
11. Using `set_constraints!` to add or modify constraints and their Jacobian.
"""

    
    using Sequoia
    
    # Example 1: Minimal Initialization with Just `nvar`
    """
    This example demonstrates how to initialize a `SEQUOIA_pb` problem with only the number of variables specified.

    # Usage:
        example_minimal_initialization()

    # Expected Output:
        Problem initialized with default settings: 
        SEQUOIA_pb(2, [0.0, 0.0], true, nothing, nothing, nothing, nothing, Int64[], Int64[], SEQUOIA_Settings(:QPM, :LBFGS, false, 1.0e-6, 1000, 300.0, 1.0e-6, :GradientNorm, nothing, nothing, false, nothing, nothing, nothing, nothing), SEQUOIA_History(SEQUOIA_Solution_step[]), nothing)
    """
    function example_minimal_initialization()
        pb = SEQUOIA_pb(2)  # Only specify the number of variables
        println("Problem initialized with default settings: ")
        println(pb)
    end

    # Example 2: Basic Optimization Problem
    """
    This example demonstrates how to initialize a `SEQUOIA_pb` optimization problem 
    with a simple quadratic objective function. No gradient, constraints, or Jacobian are provided.
    
    # Usage:
        example_basic_problem()
    
    # Expected Output:
        Objective function set. Problem initialized.
        SEQUOIA_pb(3, [0.0, 0.0, 0.0], true, var"#35#36"(), nothing, nothing, nothing, Int64[], Int64[], SEQUOIA_Settings(:QPM, :LBFGS, false, 1.0e-6, 1000, 300.0, 1.0e-6, :GradientNorm, nothing, nothing, false, nothing, nothing, nothing, nothing), SEQUOIA_History(SEQUOIA_Solution_step[]), nothing)
        # The above is a SEQUOIA_pb instance with a quadratic objective and default settings.
    """
    function example_basic_problem()
        pb = SEQUOIA_pb(
            3;
            x0=[0.0, 0.0, 0.0],
            objective=x -> sum(x.^2)
        )
        println("Objective function set. Problem initialized.")
        println(pb)
    end
    
    # Example 3: Optimization Problem with Provided Gradient
    """
    This example shows how to define an optimization problem with both an objective function 
    and its explicitly provided gradient.
    
    # Usage:
        example_gradient_provided()
    
    # Expected Output:
        Gradient function provided:
        [2.0, 4.0, 6.0]
    """
    function example_gradient_provided()
        pb = SEQUOIA_pb(
            3;
            x0=[1.0, 2.0, 3.0],
            objective=x -> sum(x.^2),
            gradient=x -> 2 .* x
        )
        println("Gradient function provided: ")
        println(pb.gradient([1.0, 2.0, 3.0]))
    end
    
    # Example 4: Automatic Differentiation for Gradient
    """
    This example demonstrates how `SEQUOIA_pb` uses automatic differentiation to compute the gradient 
    when it is not explicitly provided.
    
    # Usage:
        example_autodiff_gradient()
    
    # Expected Output:
        Before validation, gradient is nothing: nothing
        ┌ Warning: A gradient is required. Setting one using Automatic Differentiation with ForwardDiff.
        └ @ Sequoia ~/Sequoia.jl/src/checks/Sequoia_validation.jl:121
        ┌ Warning: No constraints are set. Ensure this is intended, as SEQUOIA is tailored for constrained optimization.
        └ @ Sequoia ~/Sequoia.jl/src/checks/Sequoia_validation.jl:150
        After validation, gradient is set: #10
    """
    function example_autodiff_gradient()
        pb = SEQUOIA_pb(
            3;
            x0=[1.0, 1.0, 1.0],
            objective=x -> sum(x.^2)  # No gradient provided
        )
        println("Before validation, gradient is nothing: ", pb.gradient)
        validate_pb!(pb)  # Automatic differentiation is triggered here
        println("After validation, gradient is set: ", pb.gradient)
    end
    
    # Example 5: Adding Constraints and Jacobian
    """
    This example illustrates how to add constraints and explicitly define a Jacobian matrix 
    to a `SEQUOIA_pb` problem.
    
    # Usage:
        example_constraints()
    
    # Expected Output:
        Constraints and Jacobian set. Problem initialized.
        SEQUOIA_pb(3, [1.0, 2.0, 3.0], true, var"#43#46"(), nothing, var"#44#47"(), var"#45#48"(), [1], [2], SEQUOIA_Settings(:QPM, :LBFGS, false, 1.0e-6, 1000, 300.0, 1.0e-6, :GradientNorm, nothing, nothing, false, nothing, nothing, nothing, nothing), SEQUOIA_History(SEQUOIA_Solution_step[]), nothing)
        # The above is a problem instance with constraints and their Jacobian set.
    """
    function example_constraints()
        pb = SEQUOIA_pb(
            3;
            x0=[1.0, 2.0, 3.0],
            objective=x -> sum(x.^2),
            constraints=x -> [x[1] - 1, x[2] - 2],
            jacobian=x -> [1.0 0.0 0.0; 0.0 1.0 0.0],
            eqcon=[1],
            ineqcon=[2]
        )
        println("Constraints and Jacobian set. Problem initialized.")
        println(pb)
    end
    
    # Example 6: Automatic Differentiation for Jacobian
    """
    This example demonstrates how `SEQUOIA_pb` uses automatic differentiation to compute the Jacobian 
    when it is not explicitly provided.
    
    # Usage:
        example_autodiff_jacobian()
    
    # Expected Output:
        Before validation, Jacobian is nothing: nothing
        ┌ Warning: A gradient is required. Setting one using Automatic Differentiation with ForwardDiff.
        └ @ Sequoia ~/Sequoia.jl/src/checks/Sequoia_validation.jl:121
        ┌ Warning: A Jacobian is required. Setting one using Automatic Differentiation with ForwardDiff.
        └ @ Sequoia ~/Sequoia.jl/src/checks/Sequoia_validation.jl:184
        After validation, Jacobian is set.
        [1.0 0.0 0.0; 0.0 1.0 0.0]
    """
    function example_autodiff_jacobian()
        pb = SEQUOIA_pb(
            3;
            x0=[1.0, 2.0, 3.0],
            objective=x -> sum(x.^2),
            constraints=x -> [x[1] - 1, x[2] - 2],  # No Jacobian provided
            eqcon=[1],
            ineqcon=[2]
        )
        println("Before validation, Jacobian is nothing: ", pb.jacobian)
        validate_pb!(pb)  # Automatic differentiation is triggered here
        println("After validation, Jacobian is set.")
        println(pb.jacobian([1.0, 2.0, 3.0]))
    end
    
    # Example 7: Updating Solver Settings
    """
    This example shows how to modify the solver settings for a `SEQUOIA_pb` optimization problem.
    
    # Usage:
        example_solver_settings()
    
    # Expected Output:
        Default solver settings: SEQUOIA_Settings(:QPM, :LBFGS, false, 1.0e-6, 1000, 300.0, 1.0e-6, :GradientNorm, nothing, nothing, false, nothing, nothing, nothing, nothing)
        Updated solver settings: SEQUOIA_Settings(:QPM, :Newton, false, 1.0e-8, 500, 60.0, 1.0e-10, :GradientNorm, nothing, nothing, false, nothing, nothing, nothing, nothing)
    """
    function example_solver_settings()
        pb = SEQUOIA_pb(
            3;
            x0=[1.0, 2.0, 3.0],
            objective=x -> sum(x.^2)
        )
        println("Default solver settings: ", pb.solver_settings)
        new_settings = SEQUOIA_Settings(:QPM, :Newton, false, 1e-8, 500, 60.0, 10^-10)
        set_solver_settings!(pb, new_settings)
        println("Updated solver settings: ", pb.solver_settings)
    end
    
    # Example 8: Resetting Solution History
    """
    This example demonstrates how to reset the solution history of a `SEQUOIA_pb` problem.
    
    # Usage:
        example_reset_history()
    
    # Expected Output:
        Before resetting, solution history length: 1
        After resetting, solution history length: 0
    """
    function example_reset_history()
        pb = SEQUOIA_pb(
            3;
            x0=[1.0, 2.0, 3.0],
            objective=x -> sum(x.^2),
            gradient=x -> 2 * x
        )

        # Add a mock solution step to the history
        solution_step = SEQUOIA_Solution_step(
            1,
            0.01,
            :first_order,
            0.1,
            5,
            [1.0, 2.0, 3.0],
            14.0,
            [2.0, 4.0, 6.0]
        )
        add_iterate!(pb.solution_history, solution_step)

        println("Before resetting, solution history length: ", length(pb.solution_history.iterates))
        reset_solution_history!(pb)
        println("After resetting, solution history length: ", length(pb.solution_history.iterates))
    end

    # Example 9: Using `set_initial_guess!`
    """
    This example demonstrates how to use the `set_initial_guess!` function to modify the initial guess for variables.

    # Usage:
        example_set_initial_guess()

    # Expected Output:
        Before updating initial guess: [0.0, 0.0, 0.0]
        After updating initial guess: [2.0, 3.0, 4.0]
    """
    function example_set_initial_guess()
        pb = SEQUOIA_pb(3; objective=x -> sum(x.^2))
        println("Before updating initial guess: ", pb.x0)
        set_initial_guess!(pb, [2.0, 3.0, 4.0])
        println("After updating initial guess: ", pb.x0)
    end

    # Example 10: Using `set_objective!`
    """
    This example demonstrates how to use the `set_objective!` function to modify the objective function and gradient.

    # Usage:
        example_set_objective()

    # Expected Output:
        Before updating objective: 6.0
        After updating objective: 3.0
        New gradient: [1.0, 1.0, 1.0]
    """
    function example_set_objective()
        pb = SEQUOIA_pb(3; objective=x -> sum(x.^2))
        println("Before updating objective: ", pb.objective([1.0, 2.0, 3.0]))
        set_objective!(pb, x -> sum(x), gradient=x -> ones(length(x)))
        println("After updating objective: ", pb.objective([1.0, 2.0, 3.0]))
        println("New gradient: ", pb.gradient([1.0, 2.0, 3.0]))
    end

    # Example 11: Using `set_constraints!`
    """
    This example demonstrates how to use the `set_constraints!` function to add or modify constraints and their Jacobian.

    # Usage:
        example_set_constraints()

    # Expected Output:
        Before setting constraints: nothing
        After setting constraints: [0.0, 0.0]
    """
    function example_set_constraints()
        pb = SEQUOIA_pb(3; objective=x -> sum(x.^2))
        println("Before setting constraints: ", pb.constraints)
        set_constraints!(pb, x -> [x[1] - 1, x[2] - 2], [1], [2], jacobian=x -> [1.0 0.0 0.0; 0.0 1.0 0.0])
        println("After setting constraints: ", pb.constraints([1.0, 2.0, 3.0]))
    end
    
    # Call all examples
    example_minimal_initialization()
    example_basic_problem()
    example_gradient_provided()
    example_autodiff_gradient()
    example_constraints()
    example_autodiff_jacobian()
    example_solver_settings()
    example_reset_history()
    example_set_initial_guess()
    example_set_objective()
    example_set_constraints()

    