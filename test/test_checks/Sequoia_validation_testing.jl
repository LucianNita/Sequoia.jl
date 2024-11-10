@testset "SEQUOIA_pb Validation Tests" begin
    # Test 1: Valid SEQUOIA_pb instance (no constraints)
    valid_problem = SEQUOIA_pb(
        3;
        x0=[1.0, 2.0, 3.0],
        objective=x -> sum(x.^2),
        gradient=x -> 2 .* x
    )
    @test validate_pb!(valid_problem) === nothing  # Should not throw any errors

    # Test 2: Invalid nvar (non-positive)
    @test_throws ArgumentError invalid_nvar_problem = SEQUOIA_pb(
        0;
        x0=[1.0],
        objective=x -> sum(x.^2),
        gradient=x -> 2 .* x
    )

    # Test 3: Invalid x0 length
    @test_throws ArgumentError invalid_x0_problem = SEQUOIA_pb(
        3;
        x0=[1.0, 2.0],  # Incorrect length
        objective=x -> sum(x.^2),
        gradient=x -> 2 * x
    )

    # Test 4: Missing objective function
    no_objective_problem = SEQUOIA_pb(
        3;
        x0=[1.0, 2.0, 3.0],
        objective=nothing  # Objective is missing
    )
    @test_throws ArgumentError validate_pb!(no_objective_problem)

    # Test 5: Objective function returns invalid type
    invalid_objective_problem = SEQUOIA_pb(
        3;
        x0=[1.0, 2.0, 3.0],
        objective=x -> [sum(x)]  # Returns a vector, not a scalar
    )
    @test_throws ArgumentError validate_pb!(invalid_objective_problem)

    # Test 6: Gradient returns invalid size
    invalid_gradient_problem = SEQUOIA_pb(
        3;
        x0=[1.0, 2.0, 3.0],
        objective=x -> sum(x.^2),
        gradient=x -> [2.0, 2.0]  # Incorrect size
    )
    @test_throws ArgumentError validate_pb!(invalid_gradient_problem)

    # Test 7: Automatic differentiation for missing gradient
    autodiff_problem = SEQUOIA_pb(
        3;
        x0=[1.0, 2.0, 3.0],
        objective=x -> sum(x.^2)  # No gradient provided
    )
    validate_pb!(autodiff_problem)
    @test autodiff_problem.gradient !== nothing  # Gradient should be automatically set
    @test autodiff_problem.gradient([1.0, 2.0, 3.0]) == [2.0, 4.0, 6.0]

    # Test 8: Missing constraints function
    no_constraints_problem = SEQUOIA_pb(
        3;
        x0=[1.0, 2.0, 3.0],
        objective=x -> sum(x.^2),
        gradient=x -> 2 .* x,
        constraints=nothing
    )
    @test validate_pb!(no_constraints_problem) === nothing  # Should only emit a warning

    # Test 9: Constraints and indices mismatch
    mismatch_constraints_problem = SEQUOIA_pb(
        3;
        x0=[1.0, 2.0, 3.0],
        objective=x -> sum(x.^2),
        gradient=x -> 2 .* x,
        constraints=x -> [x[1] - 1, x[2] - 2],
        eqcon=[1],  # Only one index provided
        ineqcon=Int[]  # Missing the second index
    )
    @test_throws ArgumentError validate_pb!(mismatch_constraints_problem)

    # Test 10: Automatic differentiation for missing Jacobian
    autodiff_jacobian_problem = SEQUOIA_pb(
        3;
        x0=[1.0, 2.0, 3.0],
        objective=x -> sum(x.^2),
        gradient=x -> 2 .* x,
        constraints=x -> [x[1] - 1, x[2] - 2],  # Constraints defined, but no Jacobian
        eqcon=[1],                              # First constraint is x[1] - 1 = 0
        ineqcon=[2]                             # Second constraint is x[2] - 2 ≤ 0
    )
    validate_pb!(autodiff_jacobian_problem)
    @test autodiff_jacobian_problem.jacobian !== nothing  # Jacobian should be set
    jacobian_result = autodiff_jacobian_problem.jacobian([1.0, 2.0, 3.0])
    @test size(jacobian_result) == (2, 3)  # Correct size for Jacobian

    # Test 11: Valid problem with constraints and Jacobian
    valid_constraints_problem = SEQUOIA_pb(
        3;
        x0=[1.0, 2.0, 3.0],
        objective=x -> sum(x.^2),
        gradient=x -> 2 .* x,
        constraints=x -> [x[1] - 1, x[2] - 2],
        jacobian=x -> [1.0 0.0 0.0; 0.0 1.0 0.0],
        eqcon=[1],
        ineqcon=[2],
    )
    @test validate_pb!(valid_constraints_problem) === nothing  # Should pass without errors
    
    # Test 12: Solver settings fallback
    problem = SEQUOIA_pb(3)
    invalid_settings = "InvalidSettings"  # Invalid type for solver settings
    @test_throws ArgumentError set_solver_settings!(problem, invalid_settings)

    # Test 13: Fallback method for non-SEQUOIA_pb object
    @test_throws ArgumentError set_objective!("not_a_problem", x -> sum(x.^2))  # Should trigger `pb_fallback`

    # Test 14: Fallback for non-function objective
    invalid_problem = SEQUOIA_pb(
        3;
        x0=[1.0, 2.0, 3.0]
    )
    @test_throws ArgumentError set_objective!(invalid_problem, "not_a_function")  # Should trigger `objective_setter_fallback`

    # Test 15: Invalid Jacobian size 
    invalid_jacobian_problem = SEQUOIA_pb(
        3;
        x0=[1.0, 2.0, 3.0],
        objective=x -> sum(x.^2),
        gradient=x -> 2 .* x,
        constraints=x -> [x[1] - 1, x[2] - 2],
        jacobian=x -> [1.0 0.0; 0.0 1.0],  # Incorrect Jacobian size
        eqcon=[1],
        ineqcon=[2]
    )
    @test_throws ArgumentError validate_pb!(invalid_jacobian_problem)  # Should fail due to incorrect Jacobian size

    # Test 16: Mismatched equality and inequality indices
    invalid_indices_problem = SEQUOIA_pb(
        3;
        x0=[1.0, 2.0, 3.0],
        objective=x -> sum(x.^2),
        gradient=x -> 2 .* x,
        constraints=x -> [x[1] - 1, x[2] - 2, x[3] - 3],  # Three constraints defined
        eqcon=[1],                                        # Only index 1 specified for equality
        ineqcon=[2,2]                                     # Index 3 is missing
    )
    @test_throws ArgumentError validate_pb!(invalid_indices_problem)  # Should fail due to invalid indices
end
