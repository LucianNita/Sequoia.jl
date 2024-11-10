@testset "SEQUOIA_pb Unit Tests" begin
    
    # Test 1: Minimal Initialization with Just `nvar`
    @testset "Minimal Initialization" begin
        pb = SEQUOIA_pb(2)
        @test pb.nvar == 2
        @test pb.x0 == [0.0, 0.0]
        @test pb.objective === nothing
        @test pb.gradient === nothing
        @test pb.constraints === nothing
        @test pb.jacobian === nothing
        @test pb.eqcon == Int[]
        @test pb.ineqcon == Int[]
    end
    
    # Test 2: Basic Optimization Problem
    @testset "Basic Optimization Problem" begin
        pb = SEQUOIA_pb(
            3;
            x0=[0.0, 0.0, 0.0],
            objective=x -> sum(x.^2)
        )
        @test pb.nvar == 3
        @test pb.x0 == [0.0, 0.0, 0.0]
        @test pb.is_minimization == true
        @test pb.objective([1.0, 2.0, 3.0]) == 14.0
        @test pb.gradient === nothing
        @test pb.constraints === nothing
        @test pb.jacobian === nothing
        @test pb.eqcon == Int[]
        @test pb.ineqcon == Int[]
        @test pb.solution_history.iterates == []
    end

    # Test 3: Optimization Problem with Provided Gradient
    @testset "Optimization Problem with Provided Gradient" begin
        pb = SEQUOIA_pb(
            3;
            x0=[1.0, 2.0, 3.0],
            objective=x -> sum(x.^2),
            gradient=x -> 2 .* x
        )
        @test pb.nvar == 3
        @test pb.x0 == [1.0, 2.0, 3.0]
        @test pb.gradient !== nothing
        @test pb.gradient([1.0, 2.0, 3.0]) == [2.0, 4.0, 6.0]
        @test pb.constraints === nothing
        @test pb.jacobian === nothing
        @test pb.eqcon == Int[]
        @test pb.ineqcon == Int[]
    end

    # Test 4: Automatic Differentiation for Gradient
    @testset "Automatic Differentiation for Gradient" begin
        pb = SEQUOIA_pb(
            3;
            x0=[1.0, 1.0, 1.0],
            objective=x -> sum(x.^2)
        )
        @test pb.gradient === nothing
        validate_pb!(pb)
        @test pb.gradient !== nothing
        @test pb.gradient([1.0, 1.0, 1.0]) == [2.0, 2.0, 2.0]
        @test pb.constraints === nothing
        @test pb.jacobian === nothing
    end

    # Test 5: Adding Constraints and Jacobian
    @testset "Adding Constraints and Jacobian" begin
        pb = SEQUOIA_pb(
            3;
            x0=[1.0, 2.0, 3.0],
            objective=x -> sum(x.^2),
            constraints=x -> [x[1] - 1, x[2] - 2],
            jacobian=x -> [1.0 0.0 0.0; 0.0 1.0 0.0],
            eqcon=[1],
            ineqcon=[2]
        )
        @test pb.constraints !== nothing
        @test pb.constraints([1.0, 2.0, 3.0]) == [0.0, 0.0]
        @test pb.jacobian !== nothing
        @test pb.jacobian([1.0, 2.0, 3.0]) == [1.0 0.0 0.0; 0.0 1.0 0.0]
        @test pb.eqcon == [1]
        @test pb.ineqcon == [2]
        @test pb.x0 == [1.0, 2.0, 3.0]
    end

    # Test 6: Automatic Differentiation for Jacobian
    @testset "Automatic Differentiation for Jacobian" begin
        pb = SEQUOIA_pb(
            3;
            x0=[1.0, 2.0, 3.0],
            objective=x -> sum(x.^2),
            constraints=x -> [x[1] - 1, x[2] - 2],
            eqcon=[1],
            ineqcon=[2]
        )
        @test pb.jacobian === nothing
        validate_pb!(pb)
        @test pb.jacobian !== nothing
        @test size(pb.jacobian([1.0, 2.0, 3.0])) == (2, 3)
        @test pb.constraints !== nothing
        @test pb.constraints([1.0, 2.0, 3.0]) == [0.0, 0.0]
    end

    # Test 7: Updating Solver Settings
    @testset "Updating Solver Settings" begin
        pb = SEQUOIA_pb(
            3;
            x0=[1.0, 2.0, 3.0],
            objective=x -> sum(x.^2)
        )
        @test pb.solver_settings.outer_method == :QPM
        @test pb.solver_settings.inner_solver == :LBFGS
        new_settings = SEQUOIA_Settings(:QPM, :Newton, false, 1e-8, 500, 60.0, 1e-10)
        set_solver_settings!(pb, new_settings)
        @test pb.solver_settings.outer_method == :QPM
        @test pb.solver_settings.inner_solver == :Newton
        @test pb.solver_settings.resid_tolerance == 1e-8
    end

    # Test 8: Resetting Solution History
    @testset "Resetting Solution History" begin
        pb = SEQUOIA_pb(
            3;
            x0=[1.0, 2.0, 3.0],
            objective=x -> sum(x.^2),
            gradient=x -> 2 .* x
        )
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
        @test length(pb.solution_history.iterates) == 1
        @test pb.solution_history.iterates[1].fval == 14.0
        reset_solution_history!(pb)
        @test length(pb.solution_history.iterates) == 0
    end

    # Test 9: Using `set_initial_guess!`
    @testset "Using set_initial_guess!" begin
        pb = SEQUOIA_pb(3; objective=x -> sum(x.^2))
        @test pb.x0 == [0.0, 0.0, 0.0]
        set_initial_guess!(pb, [2.0, 3.0, 4.0])
        @test pb.x0 == [2.0, 3.0, 4.0]
    end

    # Test 10: Using `set_objective!`
    @testset "Using set_objective!" begin
        pb = SEQUOIA_pb(3; objective=x -> sum(x.^2))
        @test pb.objective([1.0, 2.0, 3.0]) == 14.0
        set_objective!(pb, x -> sum(x), gradient=x -> ones(length(x)))
        @test pb.objective([1.0, 2.0, 3.0]) == 6.0
        @test pb.gradient([1.0, 2.0, 3.0]) == [1.0, 1.0, 1.0]
    end

    # Test 11: Using `set_constraints!`
    @testset "Using set_constraints!" begin
        pb = SEQUOIA_pb(3; objective=x -> sum(x.^2))
        @test pb.constraints === nothing
        set_constraints!(pb, x -> [x[1] - 1, x[2] - 2], [1], [2], jacobian=x -> [1.0 0.0 0.0; 0.0 1.0 0.0])
        @test pb.constraints([1.0, 2.0, 3.0]) == [0.0, 0.0]
        @test pb.jacobian([1.0, 2.0, 3.0]) == [1.0 0.0 0.0; 0.0 1.0 0.0]
        @test pb.eqcon == [1]
        @test pb.ineqcon == [2]
    end
end
