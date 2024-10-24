@testset "Sequoia validation functions testing" begin

    @testset "SEQUOIA Validation Tests" begin
    
        # Test 1: Validate number of variables
        @testset "Validation of Number of Variables" begin
            # Valid number of variables
            pb_valid = SEQUOIA_pb(3, objective=x -> 1.0, x0 = [1.0, 2.0, 3.0])
            validate_pb(pb_valid)  # Should pass without errors
    
            # Invalid number of variables (should throw error)
            @test_throws ArgumentError SEQUOIA_pb(0)  # Should throw because nvar is not positive
        end
    
        # Test 2: Validate initial guess length
        @testset "Validation of Initial Guess" begin
            pb = SEQUOIA_pb(2, objective=x -> 1.0, x0 = [0.5, 0.5])
            validate_pb(pb)  # Should pass validation
    
            # Mismatched length between initial guess and number of variables
            pb_invalid_x0 = SEQUOIA_pb(3, objective=x -> 1.0, x0 = [1.0, 2.0])
            @test_throws ArgumentError validate_pb(pb_invalid_x0)  # Should throw error
        end
    
        # Test 3: Validate objective function behavior
        @testset "Validation of Objective Function" begin
            # Valid objective function
            objective_fn = x -> sum(x.^2)
            pb_valid_obj = SEQUOIA_pb(2, x0 = [1.0, 2.0], objective = objective_fn)
            validate_pb(pb_valid_obj)  # Should pass validation
    
            # Objective function returns a vector (invalid)
            invalid_objective_fn = x -> [x[1] + x[2]]
            pb_invalid_obj = SEQUOIA_pb(2, x0 = [1.0, 2.0], objective = invalid_objective_fn)
            @test_throws ArgumentError validate_pb(pb_invalid_obj)  # Should throw error
        end
    
        # Test 4: Validate gradient function
        @testset "Validation of Gradient Function" begin
            # Valid objective function, should auto-compute gradient using ForwardDiff
            objective_fn = x -> (x[1] - 3.0)^2 + (x[2] - 2.0)^2
            pb = SEQUOIA_pb(2, x0 = [1.0, 2.0], objective = objective_fn)
            validate_pb(pb)  # Should pass, with gradient auto-computed
    
            # Check that the gradient is not `nothing`
            @test pb.gradient !== nothing
            @test pb.gradient([1.0, 2.0]) ≈ [-4.0, 0.0]
    
            # Custom gradient
            custom_gradient = x -> [2 * (x[1] - 3.0), 2 * (x[2] - 2.0)]
            pb_custom_grad = SEQUOIA_pb(2, x0 = [1.0, 2.0], objective = objective_fn, gradient = custom_gradient)
            validate_pb(pb_custom_grad)  # Should pass validation with custom gradient
    
            # Invalid gradient that returns the wrong size
            invalid_gradient = x -> [x[1] - 3.0]
            pb_invalid_grad = SEQUOIA_pb(2, x0 = [1.0, 2.0], objective = objective_fn, gradient = invalid_gradient)
            @test_throws ArgumentError validate_pb(pb_invalid_grad)  # Should throw error
        end
    
        # Test 5: Validate constraints and Jacobian
        @testset "Validation of Constraints and Jacobian" begin
            # No constraints provided
            pb_no_constraints = SEQUOIA_pb(2, objective=x -> 1.0, x0 = [0.5, 0.5])
            validate_pb(pb_no_constraints)  # Should pass (no constraints is valid)
    
            # Valid constraints with correct equality/inequality specification
            constraints_fn = x -> [x[1] + x[2] - 1.0]
            pb_valid_constraints = SEQUOIA_pb(2, objective=x -> 1.0, x0 = [0.5, 0.5], constraints = constraints_fn, eqcon = [1], ineqcon = Int[])
            validate_pb(pb_valid_constraints)  # Should pass validation
    
            # Incorrect constraints specification (constraint count mismatch)
            pb_invalid_constraints = SEQUOIA_pb(2, objective=x -> 1.0, x0 = [0.5, 0.5], constraints = constraints_fn, eqcon = [1, 2], ineqcon = Int[])
            @test_throws ArgumentError validate_pb(pb_invalid_constraints)  # Should throw error
    
            # Valid constraints with auto-computed Jacobian
            pb_auto_jacobian = SEQUOIA_pb(2, objective=x -> 1.0, x0 = [0.5, 0.5], constraints = constraints_fn, eqcon = [1], ineqcon = Int[])
            validate_constraints!(pb_auto_jacobian)  # Should auto-compute Jacobian without error
            @test pb_auto_jacobian.jacobian !== nothing
        end
    
        # Test 6: Validate solver settings
        @testset "Validation of Solver Settings" begin
            # Valid solver settings
            pb = SEQUOIA_pb(2, objective=x -> 1.0, x0 = [0.5, 0.5])
            valid_settings = SEQUOIA_Settings(:QPM, :LBFGS, false, 1e-6, 1000, 3000.0)
            pb.solver_settings = valid_settings
            validate_pb(pb)  # Should pass validation
        end
    
        # Test 7: Validate Exit Code
        @testset "Validation of Exit Codes" begin
            pb = SEQUOIA_pb(2, objective=x -> 1.0, x0 = [0.5, 0.5])
    
            # Valid exit code
            pb.exitCode = :OptimalityReached
            validate_pb(pb)  # Should pass validation
    
            # Invalid exit code
            pb_invalid_code = SEQUOIA_pb(2, objective=x -> 1.0, x0 = [0.5, 0.5])
            pb_invalid_code.exitCode = :InvalidCode
            @test_throws ArgumentError validate_pb(pb_invalid_code)  # Should throw error
        end
    
    end
    
end